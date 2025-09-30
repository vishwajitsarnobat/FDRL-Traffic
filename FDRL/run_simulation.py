import os
import sys
import traci

# --- Configuration ---
USE_GUI = True
if USE_GUI:
    SUMO_BINARY = "sumo-gui"
else:
    SUMO_BINARY = "sumo"

CONFIG_FILE = "sumo_files/simulated/cross.sumocfg"

# --- Main Simulation Logic ---
def run_simulation():
    # Set up SUMO environment
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")

    # Command to start SUMO
    sumo_cmd = [SUMO_BINARY, "-c", CONFIG_FILE, "--step-length", "1", "--quit-on-end"]
    
    # Start SUMO and connect with TraCI
    traci.start(sumo_cmd)

    # --- Agent Setup ---
    # Get all traffic light IDs from the simulation
    tls_ids = traci.trafficlight.getIDList()
    
    # A dictionary to manage the state of each traffic light
    # We will store the last time we switched its phase
    tls_manager = {tls_id: {'last_switch_step': 0} for tls_id in tls_ids}
    
    # The default program has Green phases at indices 0 (NS) and 2 (EW)
    green_phases = [0, 2]
    switch_interval = 20 # Switch every 20 steps/seconds

    # --- Main Agent Control Loop ---
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # Loop through each traffic light and manage it individually
        for tls_id in tls_ids:
            # Check if it's time to switch this specific traffic light
            if step - tls_manager[tls_id]['last_switch_step'] >= switch_interval:
                
                current_phase = traci.trafficlight.getPhase(tls_id)
                
                # We only switch if it's currently in a green phase
                if current_phase in green_phases:
                    # --- YOUR RL LOGIC FOR THIS SPECIFIC tls_id WOULD GO HERE ---
                    # 1. Get state for this junction (e.g., queues on lanes leading to it)
                    # 2. Choose action (0 or 2) with your model
                    # For this demo, we just switch to the other green phase
                    
                    if current_phase == 0:
                        next_green_phase = 2
                    else:
                        next_green_phase = 0
                    # --- END OF RL LOGIC ---
                    
                    # Execute the switch for this specific traffic light
                    traci.trafficlight.setPhase(tls_id, next_green_phase)
                    
                    # Update the manager with the new switch time
                    tls_manager[tls_id]['last_switch_step'] = step
        step += 1
            
    traci.close()
    sys.stdout.flush()

if __name__ == "__main__":
    run_simulation()