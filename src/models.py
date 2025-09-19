"""
models.py - Neural Network Models for FDRL Traffic Signal Control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ActorNetwork(nn.Module):
    """
    Actor network for PPO algorithm.
    Input: State vector (8 dimensions for Indian traffic)
    Output: Action probabilities (4 dimensions for 4 phases)
    """
    
    def __init__(self, state_dim: int = 8, action_dim: int = 4):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class CriticNetwork(nn.Module):
    """
    Critic network for PPO algorithm.
    Input: State vector (8 dimensions for Indian traffic)
    Output: State value (scalar)
    """
    
    def __init__(self, state_dim: int = 8):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPOAgent:
    """
    PPO Agent for traffic signal control.
    Manages both Actor and Critic networks with PPO update logic.
    """
    
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        lr_actor: float = 0.0001,
        lr_critic: float = 0.001,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = torch.device(device)
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.old_actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        
        # Initialize old actor with same weights
        self.old_actor.load_state_dict(self.actor.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Experience buffer
        self.clear_buffer()
        
    def clear_buffer(self):
        """Clear experience buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Select action based on current policy.
        Returns action and log probability.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float
    ):
        """Store transition in experience buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        
    def compute_returns(self) -> torch.Tensor:
        """Compute discounted returns for all stored transitions"""
        returns = []
        discounted_sum = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
            
        return torch.FloatTensor(returns).to(self.device)
    
    def update(self, epochs: int = 10):
        """
        Update actor and critic networks using PPO algorithm.
        """
        if len(self.states) == 0:
            return
            
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Compute returns and advantages
        returns = self.compute_returns()
        values = self.critic(states).squeeze()
        advantages = returns - values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(epochs):
            # Get current action probabilities
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values, returns)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
        # Update old actor
        self.old_actor.load_state_dict(self.actor.state_dict())
        
        # Clear buffer
        self.clear_buffer()
        
    def get_model_parameters(self) -> dict:
        """Get model parameters for federated aggregation"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }
    
    def set_model_parameters(self, parameters: dict):
        """Set model parameters from federated aggregation"""
        self.actor.load_state_dict(parameters['actor'])
        self.critic.load_state_dict(parameters['critic'])
        self.old_actor.load_state_dict(parameters['actor'])
        
    def save_model(self, path: str):
        """Save model to disk"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'old_actor': self.old_actor.state_dict()
        }, path)
        
    def load_model(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.old_actor.load_state_dict(checkpoint['old_actor'])