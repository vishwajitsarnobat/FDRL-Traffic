"""
main.py - Main entry point for FDRL Traffic Signal Control System (Indian Traffic)
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import traci

from src.models import PPOAgent, MultiIntersectionPPOAgent
from src.environment import SUMOEnvironment
from src.trainer import IndianTrafficSignalTrainer
from src.fixed_time_controller import FixedTimeController

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def run_fdrl_training(config: dict, mode: str = 'federated'):
    print(f"\n{'='*60}\nStarting {mode.upper()} TRAINING\n{'='*60}\n")
    trainer = IndianTrafficSignalTrainer(config=config, mode=mode, save_dir=config.get('save_dir', './checkpoints'))
    trainer.train(episodes=config.get('training_episodes', 200))
    trainer.save_metrics()
    trainer.evaluate(episodes=10)
    return trainer

def run_simulation(config: dict, agent=None, controller=None):
    """Generic simulation loop for both smart agent and fixed-time controller."""
    try:
        env = SUMOEnvironment(**config['environment'])
    except Exception as e:
        print(f"Error creating environment: {e}")
        return {}

    total_episodes = config.get('test_episodes', 5)
    all_metrics = []

    for episode in range(total_episodes):
        print(f"\nEpisode {episode + 1}/{total_episodes}")
        states = env.reset()
        if controller:
            controller.reset()
        
        done = False
        episode_metrics = []

        while not done:
            actions = {}
            log_probs = {}

            if agent: # Smart Agent
                actions, log_probs = agent.select_actions(states, env.intersection_ids, deterministic=True)
            elif controller: # Fixed-time Controller
                for int_id in env.intersection_ids:
                    actions[int_id] = controller.get_action()
                controller.update()

            next_states, _, done, info = env.step(actions)
            states = next_states
            episode_metrics.append(info)

        # Aggregate metrics for the episode
        if episode_metrics:
            avg_ep_metrics = {
                'avg_waiting_time': np.mean([m['avg_waiting_time'] for m in episode_metrics]),
                'avg_queue_length': np.mean([m['avg_queue_length'] for m in episode_metrics])
            }
            all_metrics.append(avg_ep_metrics)
            print(f"  Episode Complete: Avg Waiting Time: {avg_ep_metrics['avg_waiting_time']:.2f}s, Avg Queue: {avg_ep_metrics['avg_queue_length']:.2f}")

        env.close()

    if not all_metrics:
        return {}

    # Calculate overall average metrics across all episodes
    final_metrics = {
        'avg_waiting_time': np.mean([m['avg_waiting_time'] for m in all_metrics]),
        'avg_queue_length': np.mean([m['avg_queue_length'] for m in all_metrics]),
        'avg_stops': 0 # Placeholder
    }
    return final_metrics

def run_smart_signal_control(config: dict, model_path: str = None):
    print(f"\n{'='*60}\nSMART TRAFFIC SIGNAL CONTROL\n{'='*60}\n")
    
    if not model_path:
        print("Error: --model-path is required for test and compare modes.")
        return {}

    num_intersections = len(config['environment']['intersection_ids'])
    agent = MultiIntersectionPPOAgent(num_intersections=num_intersections, **config['agent'])
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        # Load global model from a federated checkpoint
        if 'global_model' in checkpoint:
            agent.agent.set_model_parameters(checkpoint['global_model'])
        # Load a standard agent model
        else:
            agent.agent.load_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return {}

    print(f"Loaded model from: {model_path}")
    avg_metrics = run_simulation(config, agent=agent)
    
    print(f"\n--- SMART CONTROL OVERALL RESULTS ---")
    print(f"Average Waiting Time: {avg_metrics.get('avg_waiting_time', 0):.2f} seconds")
    print(f"Average Queue Length: {avg_metrics.get('avg_queue_length', 0):.2f} vehicles")
    print("-" * 35 + "\n")
    return avg_metrics

def run_fixed_time_control(config: dict):
    print(f"\n{'='*60}\nFIXED-TIME TRAFFIC SIGNAL CONTROL\n{'='*60}\n")
    controller = FixedTimeController(
        phase_duration=config.get('fixed_phase_duration', 30),
        yellow_duration=config['environment'].get('yellow_duration', 3)
    )
    avg_metrics = run_simulation(config, controller=controller)
    
    print(f"\n--- FIXED-TIME CONTROL OVERALL RESULTS ---")
    print(f"Average Waiting Time: {avg_metrics.get('avg_waiting_time', 0):.2f} seconds")
    print(f"Average Queue Length: {avg_metrics.get('avg_queue_length', 0):.2f} vehicles")
    print("-" * 40 + "\n")
    return avg_metrics

def compare_methods(config: dict, model_path: str = None):
    print(f"\n{'='*60}\nCOMPARING CONTROL METHODS\n{'='*60}\n")
    
    fixed_metrics = run_fixed_time_control(config)
    if not fixed_metrics:
        print("Fixed-time control failed, cannot compare.")
        return
    
    smart_metrics = run_smart_signal_control(config, model_path)
    if not smart_metrics:
        print("Smart control failed, cannot compare.")
        return

    print(f"\n{'='*60}\nCOMPARISON RESULTS\n{'='*60}\n")
    
    wait_improv = (fixed_metrics['avg_waiting_time'] - smart_metrics['avg_waiting_time']) / (fixed_metrics['avg_waiting_time'] + 1e-8) * 100
    queue_improv = (fixed_metrics['avg_queue_length'] - smart_metrics['avg_queue_length']) / (fixed_metrics['avg_queue_length'] + 1e-8) * 100
    
    print(f"Fixed-Time Control:")
    print(f"  - Avg Waiting Time: {fixed_metrics['avg_waiting_time']:.2f}s")
    print(f"  - Avg Queue Length: {fixed_metrics['avg_queue_length']:.2f}\n")
    
    print(f"Smart FDRL Control:")
    print(f"  - Avg Waiting Time: {smart_metrics['avg_waiting_time']:.2f}s")
    print(f"  - Avg Queue Length: {smart_metrics['avg_queue_length']:.2f}\n")
    
    print(f"Improvement with FDRL:")
    print(f"  - Waiting Time Reduction: {wait_improv:+.1f}%")
    print(f"  - Queue Length Reduction: {queue_improv:+.1f}%")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='FDRL-based Traffic Signal Control System')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'fixed', 'compare'], default='train')
    parser.add_argument('--training-mode', type=str, choices=['federated', 'individual', 'aggregated'], default='federated')
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at '{args.config}'")
        print("Run 'python src/setup_network.py' first to create a corrected config file.")
        sys.exit(1)
    
    config = load_config(args.config)
    
    try:
        if args.mode == 'train':
            run_fdrl_training(config, mode=args.training_mode)
        elif args.mode == 'test':
            run_smart_signal_control(config, model_path=args.model_path)
        elif args.mode == 'fixed':
            run_fixed_time_control(config)
        elif args.mode == 'compare':
            compare_methods(config, model_path=args.model_path)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if traci.isLoaded():
            traci.close()

if __name__ == "__main__":
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME environment variable is not set.")
        sys.exit(1)

    sumo_tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if sumo_tools not in sys.path:
        sys.path.append(sumo_tools)

    main()