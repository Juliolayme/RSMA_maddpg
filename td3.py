"""Centralized TD3 agent for RSMA joint-action optimization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

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
    checkpoint_dir: str = "tmp/TD3"
    checkpoint_name: str = "rsma_td3"


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


class TD3Agent:
    """Centralized TD3 agent that controls the joint action of both BSs."""

    def __init__(self, config: TD3Config) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(config.max_size, config.state_dim, config.action_dim)
        self.learn_step_cntr = 0

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

    def choose_action(self, state: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """Return a joint action optionally perturbed by Gaussian exploration noise."""
        self.actor.eval()
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        self.actor.train()
        if noise_scale > 0.0:
            action = action + np.random.normal(0.0, noise_scale, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def remember(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store one transition in replay memory."""
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self) -> None:
        """Run one TD3 update if sufficient samples are available."""
        if self.memory.mem_cntr < self.config.batch_size:
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
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic_2_optimizer.step()

        self.learn_step_cntr += 1
        if self.learn_step_cntr % self.config.update_actor_interval != 0:
            return

        actor_action = self.actor(state)
        actor_input = torch.cat([state, actor_action], dim=1)
        actor_loss = -self.critic_1(actor_input).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self._soft_update()

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
