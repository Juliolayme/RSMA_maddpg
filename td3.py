"""Centralized TD3 agent for RSMA joint-action optimization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class TD3Config:
    """Configuration container for the TD3 agent."""

    state_dim: int
    action_dim: int
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 1e-3
    batch_size: int = 128
    max_size: int = 100000
    hidden_dims: Tuple[int, int, int] = (256, 256, 128)
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5
    update_actor_interval: int = 2
    learning_starts: int = 1000
    max_grad_norm: float = 1.0
    alpha_action_indices: Tuple[int, ...] = ()
    checkpoint_dir: str = "tmp/TD3"
    checkpoint_name: str = "rsma_td3"


class OUNoise:
    """Ornstein-Uhlenbeck noise process for smoother exploration."""

    def __init__(self, action_dim: int, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2) -> None:
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.zeros(action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.state = np.zeros(self.action_dim, dtype=np.float32)

    def sample(self) -> np.ndarray:
        dx = self.theta * (-self.state) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state.astype(np.float32)


class ReplayBuffer:
    """Replay buffer for joint state-action transitions."""

    def __init__(self, max_size: int, state_dim: int, action_dim: int) -> None:
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.done_memory = np.zeros(max_size, dtype=np.float32)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.done_memory[idx] = 1.0 - float(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.next_state_memory[batch],
            self.done_memory[batch],
        )


class MLP(nn.Module):
    """Simple fully-connected network used by both actor and critic."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, int, int], output_dim: int, final_tanh: bool = False) -> None:
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        if final_tanh:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def set_output_bias(self, indices: Iterable[int], value: float) -> None:
        """Set selected output biases in the final linear layer."""
        final_layer = None
        for module in reversed(self.net):
            if isinstance(module, nn.Linear):
                final_layer = module
                break
        if final_layer is None:
            return
        with torch.no_grad():
            for idx in indices:
                if 0 <= idx < final_layer.bias.numel():
                    final_layer.bias[idx] = value


class TD3Agent:
    """Centralized TD3 agent that controls the joint action of both BSs."""

    def __init__(self, config: TD3Config) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(config.max_size, config.state_dim, config.action_dim)
        self.learn_step_cntr = 0
        self.ou_noise = OUNoise(config.action_dim)
        self.base_actor_lr = config.actor_lr
        self.base_critic_lr = config.critic_lr

        self.actor = MLP(config.state_dim, config.hidden_dims, config.action_dim, final_tanh=True).to(self.device)
        self.target_actor = MLP(config.state_dim, config.hidden_dims, config.action_dim, final_tanh=True).to(self.device)
        self.critic_1 = MLP(config.state_dim + config.action_dim, config.hidden_dims, 1).to(self.device)
        self.critic_2 = MLP(config.state_dim + config.action_dim, config.hidden_dims, 1).to(self.device)
        self.target_critic_1 = MLP(config.state_dim + config.action_dim, config.hidden_dims, 1).to(self.device)
        self.target_critic_2 = MLP(config.state_dim + config.action_dim, config.hidden_dims, 1).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=config.critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=config.critic_lr)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor.set_output_bias(config.alpha_action_indices, 0.0)
        self.target_actor.set_output_bias(config.alpha_action_indices, 0.0)

    def reset_noise(self) -> None:
        """Reset temporally correlated exploration noise."""
        self.ou_noise.reset()

    def choose_action(self, state: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """Return a joint action optionally perturbed by OU exploration noise."""
        self.actor.eval()
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        self.actor.train()
        if noise_scale > 0.0:
            action = action + noise_scale * self.ou_noise.sample()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def remember(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store one transition in replay memory."""
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self) -> None:
        """Run one TD3 update if sufficient samples are available."""
        if self.memory.mem_cntr < max(self.config.batch_size, self.config.learning_starts):
            return

        state, action, reward, next_state, done = self.memory.sample_buffer(self.config.batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            target_action = self.target_actor(next_state)
            noise = torch.clamp(
                torch.randn_like(target_action) * self.config.target_noise_std,
                -self.config.target_noise_clip,
                self.config.target_noise_clip,
            )
            target_action = torch.clamp(target_action + noise, -1.0, 1.0)
            target_input = torch.cat([next_state, target_action], dim=1)
            target_q1 = self.target_critic_1(target_input)
            target_q2 = self.target_critic_2(target_input)
            target_q = reward + self.config.gamma * torch.min(target_q1, target_q2) * done

        critic_input = torch.cat([state, action], dim=1)
        current_q1 = self.critic_1(critic_input)
        current_q2 = self.critic_2(critic_input)

        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.config.max_grad_norm)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.config.max_grad_norm)
        self.critic_2_optimizer.step()

        self.learn_step_cntr += 1
        if self.learn_step_cntr % self.config.update_actor_interval != 0:
            return

        actor_action = self.actor(state)
        actor_input = torch.cat([state, actor_action], dim=1)
        actor_loss = -self.critic_1(actor_input).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()
        self._soft_update()

    def set_learning_rates(self, actor_lr: float, critic_lr: float) -> None:
        """Update optimizer learning rates."""
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = actor_lr
        for optimizer in (self.critic_1_optimizer, self.critic_2_optimizer):
            for param_group in optimizer.param_groups:
                param_group["lr"] = critic_lr

    def _soft_update(self) -> None:
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)

    def save_models(self) -> None:
        """Save actor and critic weights."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        prefix = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_name)
        torch.save(self.actor.state_dict(), prefix + "_actor.pt")
        torch.save(self.target_actor.state_dict(), prefix + "_target_actor.pt")
        torch.save(self.critic_1.state_dict(), prefix + "_critic1.pt")
        torch.save(self.critic_2.state_dict(), prefix + "_critic2.pt")
        torch.save(self.target_critic_1.state_dict(), prefix + "_target_critic1.pt")
        torch.save(self.target_critic_2.state_dict(), prefix + "_target_critic2.pt")

    def load_models(self) -> None:
        """Load actor and critic weights from checkpoint files."""
        prefix = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_name)
        self.actor.load_state_dict(torch.load(prefix + "_actor.pt", map_location=self.device))
        self.target_actor.load_state_dict(torch.load(prefix + "_target_actor.pt", map_location=self.device))
        self.critic_1.load_state_dict(torch.load(prefix + "_critic1.pt", map_location=self.device))
        self.critic_2.load_state_dict(torch.load(prefix + "_critic2.pt", map_location=self.device))
        self.target_critic_1.load_state_dict(torch.load(prefix + "_target_critic1.pt", map_location=self.device))
        self.target_critic_2.load_state_dict(torch.load(prefix + "_target_critic2.pt", map_location=self.device))
