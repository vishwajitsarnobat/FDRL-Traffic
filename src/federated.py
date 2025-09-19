"""
federated.py - Federated Learning Components for FDRL
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import pickle
import logging


class FederatedAggregator:
    """
    Federated central coordinator for aggregating local model parameters.
    Implements FedAvg algorithm with flexible weighting strategies.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        aggregation_strategy: str = 'flexible',
        device: str = 'cpu'
    ):
        """
        Args:
            alpha: Federated learning rate (soft update parameter)
            aggregation_strategy: 'equal' or 'flexible' weighting
            device: Computing device
        """
        self.alpha = alpha
        self.aggregation_strategy = aggregation_strategy
        self.device = torch.device(device)
        self.global_model = None
        self.aggregation_count = 0
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def initialize_global_model(self, model_params: dict):
        """Initialize global model with parameters from first agent"""
        self.global_model = deepcopy(model_params)
        
    def aggregate(
        self,
        local_models: List[dict],
        local_scores: Optional[List[int]] = None
    ) -> dict:
        """
        Aggregate local model parameters using FedAvg.
        
        Args:
            local_models: List of local model parameter dictionaries
            local_scores: Performance scores for flexible weighting
            
        Returns:
            Aggregated global model parameters
        """
        if not local_models:
            return self.global_model
            
        # Initialize global model if not done
        if self.global_model is None:
            self.initialize_global_model(local_models[0])
            
        # Calculate weights
        weights = self._calculate_weights(len(local_models), local_scores)
        
        # Aggregate parameters
        new_global_model = {}
        
        for model_type in ['actor', 'critic']:
            new_params = {}
            
            # Get first model structure
            first_model = local_models[0][model_type]
            
            for param_name in first_model.keys():
                # Weighted average of parameters
                aggregated_param = None
                
                for i, local_model in enumerate(local_models):
                    param = local_model[model_type][param_name]
                    
                    if aggregated_param is None:
                        aggregated_param = weights[i] * param
                    else:
                        aggregated_param += weights[i] * param
                        
                new_params[param_name] = aggregated_param
                
            new_global_model[model_type] = new_params
            
        # Apply soft update
        if self.global_model is not None and self.alpha > 0:
            final_global_model = self._soft_update(
                self.global_model,
                new_global_model,
                self.alpha
            )
        else:
            final_global_model = new_global_model
            
        self.global_model = final_global_model
        self.aggregation_count += 1
        
        self.logger.info(f"Aggregation #{self.aggregation_count} completed with {len(local_models)} models")
        
        return final_global_model
    
    def _calculate_weights(
        self,
        num_models: int,
        scores: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Calculate aggregation weights based on strategy.
        
        Args:
            num_models: Number of local models
            scores: Performance scores for flexible weighting
            
        Returns:
            Weight vector
        """
        if self.aggregation_strategy == 'equal' or scores is None:
            # Equal weights
            return np.ones(num_models) / num_models
            
        elif self.aggregation_strategy == 'flexible':
            # Performance-based weights
            scores = np.array(scores)
            
            if scores.sum() == 0:
                # All scores are zero, use equal weights
                return np.ones(num_models) / num_models
            else:
                # Normalize scores to get weights
                weights = scores / scores.sum()
                return weights
                
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
            
    def _soft_update(
        self,
        old_model: dict,
        new_model: dict,
        alpha: float
    ) -> dict:
        """
        Apply soft update: W_new = alpha * W_old + (1 - alpha) * W_aggregated
        
        Args:
            old_model: Previous global model
            new_model: Newly aggregated model
            alpha: Soft update parameter
            
        Returns:
            Updated model
        """
        updated_model = {}
        
        for model_type in ['actor', 'critic']:
            updated_params = {}
            
            for param_name in old_model[model_type].keys():
                old_param = old_model[model_type][param_name]
                new_param = new_model[model_type][param_name]
                
                updated_param = alpha * old_param + (1 - alpha) * new_param
                updated_params[param_name] = updated_param
                
            updated_model[model_type] = updated_params
            
        return updated_model
    
    def get_global_model(self) -> dict:
        """Get current global model parameters"""
        return self.global_model
    
    def save_state(self, path: str):
        """Save aggregator state to disk"""
        state = {
            'global_model': self.global_model,
            'aggregation_count': self.aggregation_count,
            'alpha': self.alpha,
            'strategy': self.aggregation_strategy
        }
        torch.save(state, path)
        
    def load_state(self, path: str):
        """Load aggregator state from disk"""
        state = torch.load(path)
        self.global_model = state['global_model']
        self.aggregation_count = state['aggregation_count']
        self.alpha = state['alpha']
        self.aggregation_strategy = state['strategy']


class FederatedClient:
    """
    Federated client wrapper for local PPO agents.
    Manages local training and communication with aggregator.
    """
    
    def __init__(
        self,
        agent,
        client_id: str,
        update_frequency: int = 10
    ):
        """
        Args:
            agent: Local PPO agent
            client_id: Unique client identifier
            update_frequency: Local updates before federation (K)
        """
        self.agent = agent
        self.client_id = client_id
        self.update_frequency = update_frequency
        
        # Training tracking
        self.local_updates = 0
        self.positive_rewards = 0
        self.total_rewards = []
        
        # Logger
        self.logger = logging.getLogger(f"Client-{client_id}")
        
    def local_train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float
    ):
        """
        Perform local training step.
        
        Returns:
            bool: Whether ready for federated update
        """
        # Store transition
        self.agent.store_transition(state, action, reward, next_state, done, log_prob)
        
        # Track rewards for scoring
        self.total_rewards.append(reward)
        if reward > 0:
            self.positive_rewards += 1
            
        # Check if time for local update
        if done or len(self.agent.states) >= self.update_frequency:
            self.agent.update()
            self.local_updates += 1
            
            # Check if ready for federation
            if self.local_updates >= self.update_frequency:
                return True
                
        return False
    
    def get_model_for_aggregation(self) -> Tuple[dict, int]:
        """
        Get model parameters and performance score for aggregation.
        
        Returns:
            Model parameters and performance score
        """
        model_params = self.agent.get_model_parameters()
        score = self.positive_rewards  # Performance score
        
        self.logger.info(f"Client {self.client_id}: Score={score}/{len(self.total_rewards)}")
        
        # Reset counters
        self.local_updates = 0
        self.positive_rewards = 0
        self.total_rewards = []
        
        return model_params, score
    
    def update_from_global(self, global_params: dict):
        """Update local model with global parameters"""
        self.agent.set_model_parameters(global_params)
        self.logger.info(f"Client {self.client_id}: Updated from global model")
        
    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Get action from local agent"""
        return self.agent.select_action(state)


class FederatedTrainer:
    """
    Orchestrates federated training across multiple clients.
    """
    
    def __init__(
        self,
        clients: List[FederatedClient],
        aggregator: FederatedAggregator,
        communication_rounds: int = 100
    ):
        """
        Args:
            clients: List of federated clients
            aggregator: Federated aggregator
            communication_rounds: Maximum federation rounds
        """
        self.clients = clients
        self.aggregator = aggregator
        self.communication_rounds = communication_rounds
        self.current_round = 0
        
        # Metrics tracking
        self.federation_history = {
            'rounds': [],
            'avg_scores': [],
            'participation': []
        }
        
        # Logger
        self.logger = logging.getLogger("FederatedTrainer")
        
    def should_aggregate(self, ready_clients: List[str]) -> bool:
        """
        Determine if aggregation should occur.
        
        Args:
            ready_clients: IDs of clients ready for aggregation
            
        Returns:
            Whether to perform aggregation
        """
        # Aggregate if at least 50% of clients are ready
        min_clients = max(1, len(self.clients) // 2)
        return len(ready_clients) >= min_clients
    
    def aggregate_models(self, participating_clients: List[str]):
        """
        Perform federated aggregation with participating clients.
        
        Args:
            participating_clients: IDs of participating clients
        """
        if not participating_clients:
            return
            
        # Collect models and scores from participating clients
        local_models = []
        local_scores = []
        
        for client in self.clients:
            if client.client_id in participating_clients:
                model_params, score = client.get_model_for_aggregation()
                local_models.append(model_params)
                local_scores.append(score)
                
        # Perform aggregation
        global_model = self.aggregator.aggregate(local_models, local_scores)
        
        # Update all clients with global model
        for client in self.clients:
            client.update_from_global(global_model)
            
        # Track metrics
        self.current_round += 1
        self.federation_history['rounds'].append(self.current_round)
        self.federation_history['avg_scores'].append(np.mean(local_scores))
        self.federation_history['participation'].append(len(participating_clients))
        
        self.logger.info(
            f"Federation round {self.current_round}: "
            f"{len(participating_clients)} clients, "
            f"avg_score={np.mean(local_scores):.2f}"
        )
        
    def get_metrics(self) -> dict:
        """Get federated training metrics"""
        return self.federation_history