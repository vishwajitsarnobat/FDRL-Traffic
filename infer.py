# infer.py
import yaml
import torch
import numpy as np
import traci
import argparse
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

from sumo_simulator import SumoSimulator
from ppo_agent import Actor

def run_inference(config, mode):
    # --- Setup ---
    config['sumo']['gui'] = True # Always use GUI for inference
    sim = SumoSimulator(
        config['sumo']['config_file'],
        step_length=config['sumo']['step_length'],
        gui=config['sumo']['gui']
    )
    
    all_junctions = sim.junctions
    controlled_junction_ids = config['system']['controlled_junctions']
    if not controlled_junction_ids:
        controlled_junction_ids = list(all_junctions.keys())

    print(f"Starting inference in '{mode}' mode for junctions: {controlled_junction_ids}")

    # --- Mode-Specific Initialization ---
    if mode == 'rl':
        # Load the trained global model for each agent
        actor_models = {}
        for j_id in controlled_junction_ids:
            j_info = all_junctions[j_id]
            action_dim = len(j_info['incoming_roads'])
            state_dim = 2 * action_dim
            
            actor = Actor(state_dim, action_dim, config['model']['hidden_layers'])
            try:
                actor.load_state_dict(torch.load(config['system']['model_save_path']))
                actor.eval() # Set model to evaluation mode
                actor_models[j_id] = actor
                print(f"Successfully loaded model for junction {j_id}.")
            except FileNotFoundError:
                print(f"FATAL: Model file not found at {config['system']['model_save_path']}. Please train a model first.")
                sim.close()
                return
        sim.init_phase_timers(controlled_junction_ids)

    # --- Data Logging ---
    simulation_log = []

    # --- Main Simulation Loop ---
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        sim.simulation_step()
        
        if mode == 'rl':
            sim.update_phase_timers()
            for j_id in controlled_junction_ids:
                # Decide if it's time to choose a new action for this junction
                if sim.phase_timers[j_id] >= config['fdrl']['green_time']:
                    state = sim.get_state(j_id)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    
                    with torch.no_grad():
                        action_probs = actor_models[j_id](state_tensor)
                    
                    # Use deterministic action for inference
                    action = torch.argmax(action_probs).item()
                    
                    sim.set_phase_inference(j_id, action, config['fdrl']['yellow_time'])

        # Log data every second
        log_entry = {'step': step}
        for j_id in controlled_junction_ids:
            state = sim.get_state(j_id)
            queue_len = int(np.sum(state[::2]))
            wait_time = np.sum(state[1::2])
            log_entry[f'{j_id}_queue'] = queue_len
            log_entry[f'{j_id}_wait_time'] = wait_time

            # Send to REST API if enabled
            if config['api']['enabled'] and mode == 'rl':
                api_payload = {
                    "junction_id": j_id,
                    "timestamp": time.time(),
                    "queue_lengths": [int(q) for q in state[::2]],
                    "waiting_times": [float(w) for w in state[1::2]],
                    "current_action": sim.current_actions[j_id]
                }
                try:
                    requests.post(f"http://{config['api']['host']}:{config['api']['port']}{config['api']['endpoint']}", json=api_payload, timeout=0.1)
                except requests.exceptions.RequestException:
                    pass # Don't crash if API is not running
        
        simulation_log.append(log_entry)
        step += 1

    sim.close()
    print("Inference finished. Generating performance plots...")
    generate_plots(simulation_log, controlled_junction_ids, mode)

def generate_plots(log, junction_ids, mode):
    df = pd.DataFrame(log)
    
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for j_id in junction_ids:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        fig.suptitle(f"Performance for Junction '{j_id}' (Mode: {mode.upper()})", fontsize=16)

        # Plot Queue Length
        ax1.plot(df['step'], df[f'{j_id}_queue'], label='Total Queue Length', color='tab:red')
        ax1.set_ylabel("Number of Queued Vehicles")
        ax1.legend()
        
        # Plot Waiting Time
        ax2.plot(df['step'], df[f'{j_id}_wait_time'], label='Total Waiting Time', color='tab:blue')
        ax2.set_ylabel("Total Waiting Time (s)")
        ax2.set_xlabel("Simulation Step (s)")
        ax2.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_filename = os.path.join(output_dir, f"plot_{j_id}_{mode}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot to {plot_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run FDRL inference for traffic control.")
    parser.add_argument('--mode', type=str, required=True, choices=['rl', 'fixed'],
                        help="Mode to run the simulation in: 'rl' for model-based control, 'fixed' for baseline.")
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    run_inference(config, args.mode)