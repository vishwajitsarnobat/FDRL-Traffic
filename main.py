"""
main.py - Main entry point for FDRL Traffic Signal Control System
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import traci
from datetime import datetime

from src.models import PPOAgent
from src.environment import SUMOEnvironment
from src.trainer import TrafficSignalTrainer
from fixed_time_controller import FixedTimeController


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def run_fdrl_training(config: dict, mode: str = 'federated'):
    """
    Run FDRL training for traffic signal control.
    
    Args:
        config: Configuration dictionary
        mode: Training mode ('federated', 'individual', or 'aggregated')
    """
    print(f"\n{'='*60}")
    print(f"Starting {mode.upper()} TRAINING")
    print(f"{'='*60}\n")
    
    # Create trainer
    trainer = TrafficSignalTrainer(
        config=config,
        mode=mode,
        save_dir=config.get('save_dir', './checkpoints')
    )
    
    # Run training
    episodes = config.get('training_episodes', 200)
    trainer.train(episodes=episodes)
    
    # Save metrics
    trainer.save_metrics()
    
    # Evaluate performance
    eval_results = trainer.evaluate(episodes=10)
    
    print(f"\n{'='*60}")
    print(f"Training Complete - Evaluation Results:")
    print(f"Average Waiting Time: {eval_results['avg_waiting_time']:.2f} seconds")
    print(f"Average Queue Length: {eval_results['avg_queue_length']:.2f} vehicles")
    print(f"Average Stops: {eval_results['avg_stops']:.2f}")
    print(f"Total Throughput: {eval_results['total_throughput']} vehicles")
    print(f"{'='*60}\n")
    
    return trainer


def run_smart_signal_control(config: dict, model_path: str = None):
    """
    Run smart traffic signal control using trained FDRL model.
    
    Args:
        config: Configuration dictionary
        model_path: Path to trained model checkpoint
    """
    print(f"\n{'='*60}")
    print(f"SMART TRAFFIC SIGNAL CONTROL (FDRL)")
    print(f"{'='*60}\n")
    
    # Load trained model
    if model_path is None:
        # Use latest checkpoint
        checkpoint_dir = config.get('save_dir', './checkpoints')
        checkpoints = [f for f in os.listdir(checkpoint_dir) if 'checkpoint' in f]
        if not checkpoints:
            print("No trained model found! Please train a model first.")
            return
        model_path = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
    
    # Create agent and load model
    agent = PPOAgent(**config['agent'])
    
    if 'federated' in model_path:
        # Load federated model
        checkpoint = torch.load(model_path)
        agent.set_model_parameters(checkpoint['global_model'])
    else:
        # Load individual model
        agent.load_model(model_path)
    
    print(f"Loaded model from: {model_path}")
    
    # Create environment
    env = SUMOEnvironment(**config['environment'])
    
    # Run simulation
    total_episodes = config.get('test_episodes', 5)
    metrics = []
    
    for episode in range(total_episodes):
        print(f"\nEpisode {episode + 1}/{total_episodes}")
        states = env.reset()
        done = False
        step = 0
        
        while not done:
            # Get state for single intersection
            state = list(states.values())[0]
            
            # Select action using trained model
            action, _ = agent.select_action(state)
            
            # Execute action
            next_states, rewards, done, info = env.step({env.intersection_ids[0]: action})
            
            # Update states
            states = next_states
            step += 1
            
            if step % 100 == 0:
                print(f"  Step {step}: Avg Waiting Time = {info['avg_waiting_time']:.2f}s")
        
        # Collect episode metrics
        episode_metrics = env.get_metrics()
        metrics.append(episode_metrics)
        
        print(f"  Episode Complete:")
        print(f"    - Average Waiting Time: {episode_metrics['avg_waiting_time']:.2f}s")
        print(f"    - Average Queue Length: {episode_metrics['avg_queue_length']:.2f}")
        print(f"    - Average Stops: {episode_metrics['avg_stops']:.2f}")
        
        env.close()
    
    # Calculate overall metrics
    avg_metrics = {
        'avg_waiting_time': np.mean([m['avg_waiting_time'] for m in metrics]),
        'avg_queue_length': np.mean([m['avg_queue_length'] for m in metrics]),
        'avg_stops': np.mean([m['avg_stops'] for m in metrics]),
        'total_throughput': sum([m['total_throughput'] for m in metrics])
    }
    
    print(f"\n{'='*60}")
    print(f"SMART SIGNAL CONTROL - Overall Results:")
    print(f"Average Waiting Time: {avg_metrics['avg_waiting_time']:.2f} seconds")
    print(f"Average Queue Length: {avg_metrics['avg_queue_length']:.2f} vehicles")
    print(f"Average Stops: {avg_metrics['avg_stops']:.2f}")
    print(f"Total Throughput: {avg_metrics['total_throughput']} vehicles")
    print(f"{'='*60}\n")
    
    return avg_metrics


def run_fixed_time_control(config: dict):
    """
    Run traditional fixed-time traffic signal control.
    
    Args:
        config: Configuration dictionary
    """
    print(f"\n{'='*60}")
    print(f"FIXED-TIME TRAFFIC SIGNAL CONTROL")
    print(f"{'='*60}\n")
    
    # Create controller
    controller = FixedTimeController(
        phase_duration=config.get('fixed_phase_duration', 30),
        yellow_duration=config.get('yellow_duration', 3)
    )
    
    # Create environment
    env = SUMOEnvironment(**config['environment'])
    
    # Run simulation
    total_episodes = config.get('test_episodes', 5)
    metrics = []
    
    for episode in range(total_episodes):
        print(f"\nEpisode {episode + 1}/{total_episodes}")
        states = env.reset()
        done = False
        step = 0
        controller.reset()
        
        while not done:
            # Get fixed-time action
            action = controller.get_action()
            
            # Execute action
            next_states, rewards, done, info = env.step({env.intersection_ids[0]: action})
            
            # Update controller
            controller.update()
            
            # Update states
            states = next_states
            step += 1
            
            if step % 100 == 0:
                print(f"  Step {step}: Avg Waiting Time = {info['avg_waiting_time']:.2f}s")
        
        # Collect episode metrics
        episode_metrics = env.get_metrics()
        metrics.append(episode_metrics)
        
        print(f"  Episode Complete:")
        print(f"    - Average Waiting Time: {episode_metrics['avg_waiting_time']:.2f}s")
        print(f"    - Average Queue Length: {episode_metrics['avg_queue_length']:.2f}")
        print(f"    - Average Stops: {episode_metrics['avg_stops']:.2f}")
        
        env.close()
    
    # Calculate overall metrics
    avg_metrics = {
        'avg_waiting_time': np.mean([m['avg_waiting_time'] for m in metrics]),
        'avg_queue_length': np.mean([m['avg_queue_length'] for m in metrics]),
        'avg_stops': np.mean([m['avg_stops'] for m in metrics]),
        'total_throughput': sum([m['total_throughput'] for m in metrics])
    }
    
    print(f"\n{'='*60}")
    print(f"FIXED-TIME CONTROL - Overall Results:")
    print(f"Average Waiting Time: {avg_metrics['avg_waiting_time']:.2f} seconds")
    print(f"Average Queue Length: {avg_metrics['avg_queue_length']:.2f} vehicles")
    print(f"Average Stops: {avg_metrics['avg_stops']:.2f}")
    print(f"Total Throughput: {avg_metrics['total_throughput']} vehicles")
    print(f"{'='*60}\n")
    
    return avg_metrics


def compare_methods(config: dict, model_path: str = None):
    """
    Compare FDRL smart control with fixed-time control.
    
    Args:
        config: Configuration dictionary
        model_path: Path to trained FDRL model
    """
    print(f"\n{'='*60}")
    print(f"COMPARING TRAFFIC SIGNAL CONTROL METHODS")
    print(f"{'='*60}\n")
    
    # Run fixed-time control
    print("Testing Fixed-Time Control...")
    fixed_metrics = run_fixed_time_control(config)
    
    # Run smart control
    print("\nTesting Smart FDRL Control...")
    smart_metrics = run_smart_signal_control(config, model_path)
    
    # Calculate improvements
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS:")
    print(f"{'='*60}")
    
    waiting_improvement = (
        (fixed_metrics['avg_waiting_time'] - smart_metrics['avg_waiting_time']) 
        / fixed_metrics['avg_waiting_time'] * 100
    )
    
    queue_improvement = (
        (fixed_metrics['avg_queue_length'] - smart_metrics['avg_queue_length'])
        / fixed_metrics['avg_queue_length'] * 100
    )
    
    stops_improvement = (
        (fixed_metrics['avg_stops'] - smart_metrics['avg_stops'])
        / fixed_metrics['avg_stops'] * 100
    )
    
    print(f"\nFixed-Time Control:")
    print(f"  - Avg Waiting Time: {fixed_metrics['avg_waiting_time']:.2f}s")
    print(f"  - Avg Queue Length: {fixed_metrics['avg_queue_length']:.2f}")
    print(f"  - Avg Stops: {fixed_metrics['avg_stops']:.2f}")
    
    print(f"\nSmart FDRL Control:")
    print(f"  - Avg Waiting Time: {smart_metrics['avg_waiting_time']:.2f}s")
    print(f"  - Avg Queue Length: {smart_metrics['avg_queue_length']:.2f}")
    print(f"  - Avg Stops: {smart_metrics['avg_stops']:.2f}")
    
    print(f"\nImprovement with FDRL:")
    print(f"  - Waiting Time Reduction: {waiting_improvement:.1f}%")
    print(f"  - Queue Length Reduction: {queue_improvement:.1f}%")
    print(f"  - Stops Reduction: {stops_improvement:.1f}%")
    
    print(f"{'='*60}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='FDRL-based Traffic Signal Control System'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'fixed', 'compare'],
        default='train',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--training-mode',
        type=str,
        choices=['federated', 'individual', 'aggregated'],
        default='federated',
        help='Training mode to use when --mode is set to "train".'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to the configuration JSON file (default: config.json)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to a trained model checkpoint for "test" or "compare" modes.'
    )

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at '{args.config}'")
        sys.exit(1)
    
    config = load_config(args.config)

    # Execute the selected mode
    if args.mode == 'train':
        run_fdrl_training(config, mode=args.training_mode)
    elif args.mode == 'test':
        run_smart_signal_control(config, model_path=args.model_path)
    elif args.mode == 'fixed':
        run_fixed_time_control(config)
    elif args.mode == 'compare':
        compare_methods(config, model_path=args.model_path)


if __name__ == "__main__":
    # Before running, ensure SUMO_HOME is set in environment variables
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME environment variable is not set. Please set it to your SUMO installation directory.")
        sys.exit(1)

    # Add the SUMO tools directory to the Python path
    sumo_tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if sumo_tools not in sys.path:
        sys.path.append(sumo_tools)

    # Run the main function
    main()