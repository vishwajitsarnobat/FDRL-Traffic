"""
trainer.py - Training orchestration for FDRL traffic signal control
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
import os

from models import PPOAgent
from environment import SUMOEnvironment
from federated import FederatedAggregator, FederatedClient, FederatedTrainer


class TrafficSignalTrainer:
    """
    Main trainer for FDRL-based traffic signal control.
    Supports both individual and federated training modes.
    """
    
    def __init__(
        self,
        config: dict,
        mode: str = 'federated',
        save_dir: str = './checkpoints'
    ):
        """
        Args:
            config: Configuration dictionary
            mode: 'individual', 'federated', or 'aggregated'
            save_dir: Directory for saving models
        """
        self.config = config
        self.mode = mode
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components based on mode
        if mode == 'federated':
            self._setup_federated_training()
        elif mode == 'individual':
            self._setup_individual_training()
        elif mode == 'aggregated':
            self._setup_aggregated_training()
        else:
            raise ValueError(f"Unknown training mode: {mode}")
            
        # Metrics tracking
        self.training_metrics = {
            'episodes': [],
            'rewards': [],
            'waiting_times': [],
            'convergence_steps': []
        }
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(
            self.save_dir, 
            f"training_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("TrafficSignalTrainer")
        
    def _setup_federated_training(self):
        """Setup components for federated training"""
        # Create multiple environments with different traffic patterns
        self.environments = []
        self.clients = []
        
        traffic_patterns = self.config.get('traffic_patterns', [
            {'ns_rate': 0.05, 'ew_rate': 0.05},  # Low saturation
            {'ns_rate': 0.05, 'ew_rate': 0.15},  # Unbalanced 1
            {'ns_rate': 0.05, 'ew_rate': 0.3},   # Unbalanced 2
            {'ns_rate': 0.15, 'ew_rate': 0.3}    # Oversaturated
        ])
        
        for i, pattern in enumerate(traffic_patterns):
            # Create environment
            env_config = self.config['environment'].copy()
            env_config['traffic_pattern'] = pattern
            env = SUMOEnvironment(**env_config)
            self.environments.append(env)
            
            # Create agent
            agent = PPOAgent(**self.config['agent'])
            
            # Create federated client
            client = FederatedClient(
                agent=agent,
                client_id=f"client_{i}",
                update_frequency=self.config['federated']['update_frequency']
            )
            self.clients.append(client)
            
        # Create aggregator
        self.aggregator = FederatedAggregator(
            alpha=self.config['federated']['alpha'],
            aggregation_strategy=self.config['federated']['aggregation_strategy']
        )
        
        # Create federated trainer
        self.fed_trainer = FederatedTrainer(
            clients=self.clients,
            aggregator=self.aggregator,
            communication_rounds=self.config['federated']['communication_rounds']
        )
        
        self.logger.info(f"Federated training setup with {len(self.clients)} clients")
        
    def _setup_individual_training(self):
        """Setup components for individual training"""
        # Single environment and agent
        self.environment = SUMOEnvironment(**self.config['environment'])
        self.agent = PPOAgent(**self.config['agent'])
        
        self.logger.info("Individual training setup complete")
        
    def _setup_aggregated_training(self):
        """Setup components for aggregated training (all data centralized)"""
        # Multiple environments but single agent
        self.environments = []
        
        traffic_patterns = self.config.get('traffic_patterns', [
            {'ns_rate': 0.05, 'ew_rate': 0.05},
            {'ns_rate': 0.05, 'ew_rate': 0.15},
            {'ns_rate': 0.05, 'ew_rate': 0.3},
            {'ns_rate': 0.15, 'ew_rate': 0.3}
        ])
        
        for pattern in traffic_patterns:
            env_config = self.config['environment'].copy()
            env_config['traffic_pattern'] = pattern
            env = SUMOEnvironment(**env_config)
            self.environments.append(env)
            
        self.agent = PPOAgent(**self.config['agent'])
        
        self.logger.info(f"Aggregated training setup with {len(self.environments)} environments")
        
    def train(self, episodes: int = 200):
        """
        Main training loop.
        
        Args:
            episodes: Number of training episodes
        """
        if self.mode == 'federated':
            self._train_federated(episodes)
        elif self.mode == 'individual':
            self._train_individual(episodes)
        elif self.mode == 'aggregated':
            self._train_aggregated(episodes)
            
    def _train_federated(self, episodes: int):
        """Federated training loop"""
        self.logger.info(f"Starting federated training for {episodes} episodes")
        
        for episode in range(episodes):
            episode_rewards = []
            episode_waiting_times = []
            ready_clients = []
            
            # Train each client in parallel (simulated)
            for i, (client, env) in enumerate(zip(self.clients, self.environments)):
                # Reset environment
                states = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    # Get action from client
                    state = list(states.values())[0]  # Single intersection
                    action, log_prob = client.get_action(state)
                    
                    # Step environment
                    next_states, rewards, done, info = env.step({env.intersection_ids[0]: action})
                    next_state = list(next_states.values())[0]
                    reward = list(rewards.values())[0]
                    
                    # Local training step
                    ready = client.local_train_step(
                        state, action, reward, next_state, done, log_prob
                    )
                    
                    if ready:
                        ready_clients.append(client.client_id)
                        
                    episode_reward += reward
                    episode_waiting_times.append(info['avg_waiting_time'])
                    states = next_states
                    
                episode_rewards.append(episode_reward)
                env.close()
                
            # Check for federated aggregation
            if self.fed_trainer.should_aggregate(ready_clients):
                self.fed_trainer.aggregate_models(ready_clients)
                ready_clients = []
                
            # Log metrics
            avg_reward = np.mean(episode_rewards)
            avg_waiting = np.mean(episode_waiting_times)
            
            self.training_metrics['episodes'].append(episode)
            self.training_metrics['rewards'].append(avg_reward)
            self.training_metrics['waiting_times'].append(avg_waiting)
            
            if episode % 10 == 0:
                self.logger.info(
                    f"Episode {episode}: "
                    f"Avg Reward={avg_reward:.2f}, "
                    f"Avg Waiting={avg_waiting:.2f}s"
                )
                
            # Save checkpoint
            if episode % 50 == 0:
                self._save_checkpoint(episode)
                
    def _train_individual(self, episodes: int):
        """Individual training loop"""
        self.logger.info(f"Starting individual training for {episodes} episodes")
        
        for episode in range(episodes):
            states = self.environment.reset()
            episode_reward = 0
            episode_waiting_times = []
            done = False
            
            while not done:
                state = list(states.values())[0]
                action, log_prob = self.agent.select_action(state)
                
                next_states, rewards, done, info = self.environment.step(
                    {self.environment.intersection_ids[0]: action}
                )
                next_state = list(next_states.values())[0]
                reward = list(rewards.values())[0]
                
                self.agent.store_transition(
                    state, action, reward, next_state, done, log_prob
                )
                
                if done or len(self.agent.states) >= 32:
                    self.agent.update()
                    
                episode_reward += reward
                episode_waiting_times.append(info['avg_waiting_time'])
                states = next_states
                
            self.environment.close()
            
            # Log metrics
            avg_waiting = np.mean(episode_waiting_times)
            
            self.training_metrics['episodes'].append(episode)
            self.training_metrics['rewards'].append(episode_reward)
            self.training_metrics['waiting_times'].append(avg_waiting)
            
            if episode % 10 == 0:
                self.logger.info(
                    f"Episode {episode}: "
                    f"Reward={episode_reward:.2f}, "
                    f"Avg Waiting={avg_waiting:.2f}s"
                )
                
            # Save checkpoint
            if episode % 50 == 0:
                self._save_checkpoint(episode)
                
    def _train_aggregated(self, episodes: int):
        """Aggregated training loop (centralized data)"""
        self.logger.info(f"Starting aggregated training for {episodes} episodes")
        
        for episode in range(episodes):
            all_transitions = []
            episode_rewards = []
            episode_waiting_times = []
            
            # Collect data from all environments
            for env in self.environments:
                states = env.reset()
                env_reward = 0
                done = False
                
                while not done:
                    state = list(states.values())[0]
                    action, log_prob = self.agent.select_action(state)
                    
                    next_states, rewards, done, info = env.step({env.intersection_ids[0]: action})
                    next_state = list(next_states.values())[0]
                    reward = list(rewards.values())[0]
                    
                    all_transitions.append((state, action, reward, next_state, done, log_prob))
                    
                    env_reward += reward
                    episode_waiting_times.append(info['avg_waiting_time'])
                    states = next_states
                    
                episode_rewards.append(env_reward)
                env.close()
                
            # Update agent with all collected data
            for transition in all_transitions:
                self.agent.store_transition(*transition)
                
            self.agent.update()
            
            # Log metrics
            avg_reward = np.mean(episode_rewards)
            avg_waiting = np.mean(episode_waiting_times)
            
            self.training_metrics['episodes'].append(episode)
            self.training_metrics['rewards'].append(avg_reward)
            self.training_metrics['waiting_times'].append(avg_waiting)
            
            if episode % 10 == 0:
                self.logger.info(
                    f"Episode {episode}: "
                    f"Avg Reward={avg_reward:.2f}, "
                    f"Avg Waiting={avg_waiting:.2f}s"
                )
                
            # Save checkpoint
            if episode % 50 == 0:
                self._save_checkpoint(episode)
                
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(
            self.save_dir,
            f"{self.mode}_checkpoint_ep{episode}.pt"
        )
        
        if self.mode == 'federated':
            # Save global model
            torch.save({
                'episode': episode,
                'global_model': self.aggregator.get_global_model(),
                'metrics': self.training_metrics
            }, checkpoint_path)
            self.aggregator.save_state(
                os.path.join(self.save_dir, f"aggregator_ep{episode}.pt")
            )
        else:
            # Save single agent model
            self.agent.save_model(checkpoint_path)
            torch.save({
                'episode': episode,
                'metrics': self.training_metrics
            }, checkpoint_path.replace('.pt', '_metrics.pt'))
            
        self.logger.info(f"Checkpoint saved at episode {episode}")
        
    def save_metrics(self):
        """Save training metrics to JSON"""
        metrics_path = os.path.join(
            self.save_dir,
            f"{self.mode}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
            
        self.logger.info(f"Metrics saved to {metrics_path}")
        
    def evaluate(self, episodes: int = 10):
        """
        Evaluate trained model performance.
        
        Args:
            episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Starting evaluation for {episodes} episodes")
        
        eval_metrics = {
            'avg_waiting_times': [],
            'avg_queue_lengths': [],
            'avg_stops': [],
            'throughput': []
        }
        
        # Get agent for evaluation
        if self.mode == 'federated':
            eval_agent = self.clients[0].agent  # Use first client's agent
        else:
            eval_agent = self.agent
            
        # Create evaluation environment
        eval_env = SUMOEnvironment(**self.config['environment'])
        
        for _ in range(episodes):
            states = eval_env.reset()
            done = False
            
            while not done:
                state = list(states.values())[0]
                action, _ = eval_agent.select_action(state)
                
                next_states, _, done, _ = eval_env.step({eval_env.intersection_ids[0]: action})
                states = next_states
                
            # Get episode metrics
            metrics = eval_env.get_metrics()
            eval_metrics['avg_waiting_times'].append(metrics['avg_waiting_time'])
            eval_metrics['avg_queue_lengths'].append(metrics['avg_queue_length'])
            eval_metrics['avg_stops'].append(metrics['avg_stops'])
            eval_metrics['throughput'].append(metrics['total_throughput'])
            
            eval_env.close()
            
        # Calculate averages
        results = {
            'avg_waiting_time': np.mean(eval_metrics['avg_waiting_times']),
            'avg_queue_length': np.mean(eval_metrics['avg_queue_lengths']),
            'avg_stops': np.mean(eval_metrics['avg_stops']),
            'total_throughput': np.sum(eval_metrics['throughput'])
        }
        
        self.logger.info(f"Evaluation results: {results}")
        
        return results