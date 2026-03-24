"""MADDPG baseline implementation for the RSMA environment."""

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
class MADDPGConfig:
    """Configuration for the MADDPG baseline."""

    obs_dim: int
    state_dim: int
    action_dim: int
    num_agents: int = 2
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 1e-3
    batch_size: int = 128
    max_size: int = 100000
    hidden_dims: Tuple[int, int, int] = (256, 256, 128)
    exploration_noise: float = 0.2
    checkpoint_dir: str = "tmp/MADDPG"
    checkpoint_name: str = "rsma_maddpg"


class ReplayBuffer:
    """Replay buffer storing local observations and global state."""

    def __init__(self, max_size: int, state_dim: int, obs_dim: int, action_dim: int, num_agents: int) -> None:
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.obs_memory = np.zeros((max_size, num_agents, obs_dim), dtype=np.float32)
        self.next_obs_memory = np.zeros((max_size, num_agents, obs_dim), dtype=np.float32)
        self.action_memory = np.zeros((max_size, num_agents, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros((max_size, num_agents), dtype=np.float32)
        self.done_memory = np.zeros(max_size, dtype=np.float32)

    def store_transition(self, obs_n, state, actions_n, rewards_n, next_obs_n, next_state, done) -> None:
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx] = state
        self.next_state_memory[idx] = next_state
        self.obs_memory[idx] = np.asarray(obs_n, dtype=np.float32)
        self.next_obs_memory[idx] = np.asarray(next_obs_n, dtype=np.float32)
        self.action_memory[idx] = np.asarray(actions_n, dtype=np.float32)
        self.reward_memory[idx] = np.asarray(rewards_n, dtype=np.float32)
        self.done_memory[idx] = 1.0 - float(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (
            self.obs_memory[batch],
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.next_obs_memory[batch],
            self.next_state_memory[batch],
            self.done_memory[batch],
        )


class MLP(nn.Module):
    """Generic multilayer perceptron."""

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


class AgentNetworks:
    """Actor-critic pair for one MADDPG agent."""

    def __init__(self, config: MADDPGConfig, agent_idx: int, device: torch.device) -> None:
        joint_action_dim = config.num_agents * config.action_dim
        self.actor = MLP(config.obs_dim, config.hidden_dims, config.action_dim, final_tanh=True).to(device)
        self.target_actor = MLP(config.obs_dim, config.hidden_dims, config.action_dim, final_tanh=True).to(device)
        self.critic = MLP(config.state_dim + joint_action_dim, config.hidden_dims, 1).to(device)
        self.target_critic = MLP(config.state_dim + joint_action_dim, config.hidden_dims, 1).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.name_prefix = os.path.join(config.checkpoint_dir, f"{config.checkpoint_name}_agent{agent_idx}")
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.name_prefix), exist_ok=True)
        torch.save(self.actor.state_dict(), self.name_prefix + "_actor.pt")
        torch.save(self.target_actor.state_dict(), self.name_prefix + "_target_actor.pt")
        torch.save(self.critic.state_dict(), self.name_prefix + "_critic.pt")
        torch.save(self.target_critic.state_dict(), self.name_prefix + "_target_critic.pt")

    def load(self, device: torch.device) -> None:
        self.actor.load_state_dict(torch.load(self.name_prefix + "_actor.pt", map_location=device))
        self.target_actor.load_state_dict(torch.load(self.name_prefix + "_target_actor.pt", map_location=device))
        self.critic.load_state_dict(torch.load(self.name_prefix + "_critic.pt", map_location=device))
        self.target_critic.load_state_dict(torch.load(self.name_prefix + "_target_critic.pt", map_location=device))


class MADDPG:
    """MADDPG baseline with centralized critics and decentralized actors."""

    def __init__(self, config: MADDPGConfig) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayBuffer(config.max_size, config.state_dim, config.obs_dim, config.action_dim, config.num_agents)
        self.agents = [AgentNetworks(config, idx, self.device) for idx in range(config.num_agents)]

    def choose_action(self, obs_n, noise_scale: float = 1.0):
        """Choose local actions for all agents."""
        actions = []
        for idx, obs in enumerate(obs_n):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.agents[idx].actor.eval()
            with torch.no_grad():
                action = self.agents[idx].actor(obs_t).squeeze(0).cpu().numpy()
            self.agents[idx].actor.train()
            noise = np.random.normal(0.0, self.config.exploration_noise * noise_scale, size=action.shape)
            actions.append(np.clip(action + noise, -1.0, 1.0).astype(np.float32))
        return actions

    def remember(self, obs_n, state, actions_n, rewards_n, next_obs_n, next_state, done) -> None:
        """Store a transition."""
        self.memory.store_transition(obs_n, state, actions_n, rewards_n, next_obs_n, next_state, done)

    def learn(self) -> None:
        """Run one MADDPG update step."""
        if self.memory.mem_cntr < self.config.batch_size:
            return

        obs, state, actions, rewards, next_obs, next_state, done = self.memory.sample_buffer(self.config.batch_size)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            target_actions = torch.cat(
                [agent.target_actor(next_obs[:, idx, :]) for idx, agent in enumerate(self.agents)],
                dim=1,
            )
            target_input = torch.cat([next_state, target_actions], dim=1)

        joint_actions = actions.view(self.config.batch_size, -1)
        critic_input = torch.cat([state, joint_actions], dim=1)

        for idx, agent in enumerate(self.agents):
            target_q = agent.target_critic(target_input)
            y = rewards[:, idx:idx + 1] + self.config.gamma * target_q * done
            current_q = agent.critic(critic_input)
            critic_loss = F.mse_loss(current_q, y)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            actor_actions = []
            for other_idx, other_agent in enumerate(self.agents):
                if other_idx == idx:
                    actor_actions.append(other_agent.actor(obs[:, other_idx, :]))
                else:
                    actor_actions.append(actions[:, other_idx, :].detach())
            actor_actions = torch.cat(actor_actions, dim=1)
            actor_input = torch.cat([state, actor_actions], dim=1)
            actor_loss = -agent.critic(actor_input).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            self._soft_update(agent.actor, agent.target_actor)
            self._soft_update(agent.critic, agent.target_critic)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.config.tau * source_param.data + (1.0 - self.config.tau) * target_param.data)

    def save_models(self) -> None:
        """Save checkpoints for all agents."""
        for agent in self.agents:
            agent.save()

    def load_models(self) -> None:
        """Load checkpoints for all agents."""
        for agent in self.agents:
            agent.load(self.device)
