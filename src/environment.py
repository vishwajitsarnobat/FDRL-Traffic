"""
environment.py - Traffic Environment Interface for SUMO
"""

import traci
import numpy as np
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET


class TrafficPhase:
    """Represents a traffic signal phase"""
    
    NORTH_SOUTH_STRAIGHT = 0
    NORTH_SOUTH_LEFT = 1
    EAST_WEST_STRAIGHT = 2  
    EAST_WEST_LEFT = 3
    
    # Indian traffic: One lane freed at a time
    # Format: (phase_id, description, green_lanes)
    PHASES = {
        0: ("North-South Straight", ["N_S"]),
        1: ("North-South Left", ["N_L", "S_L"]),
        2: ("East-West Straight", ["E_S", "W_S"]),
        3: ("East-West Left", ["E_L", "W_L"])
    }
    
    PHASE_DURATION = 30  # seconds
    YELLOW_DURATION = 3  # seconds


class IntersectionState:
    """Represents the state of a single intersection"""
    
    def __init__(self, intersection_id: str):
        self.intersection_id = intersection_id
        self.queue_lengths = np.zeros(4)  # N, S, E, W
        self.waiting_times = np.zeros(4)  # Average waiting time per direction
        self.current_phase = 0
        self.phase_timer = 0
        
    def get_state_vector(self) -> np.ndarray:
        """
        Get state vector for neural network input.
        For Indian traffic: 8 dimensions (4 queue lengths + 4 waiting times)
        """
        return np.concatenate([self.queue_lengths, self.waiting_times])
    
    def update_from_sumo(self, tls_id: str):
        """Update state from SUMO simulation"""
        # Get controlled lanes
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        
        # Reset state
        self.queue_lengths = np.zeros(4)
        self.waiting_times = np.zeros(4)
        
        # Map lanes to directions (simplified for Indian traffic)
        direction_map = {
            'N': 0, 'S': 1, 'E': 2, 'W': 3
        }
        
        for lane in controlled_lanes:
            # Extract direction from lane ID (assumes naming convention)
            direction = lane[0] if lane[0] in direction_map else 'N'
            dir_idx = direction_map[direction]
            
            # Get queue length
            self.queue_lengths[dir_idx] += traci.lane.getLastStepHaltingNumber(lane)
            
            # Get waiting time of vehicles
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            if vehicle_ids:
                wait_times = [traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids]
                self.waiting_times[dir_idx] = max(self.waiting_times[dir_idx], 
                                                  np.mean(wait_times) if wait_times else 0)


class SUMOEnvironment:
    """
    SUMO Environment wrapper for traffic signal control.
    Adapted for Indian traffic conditions.
    """
    
    def __init__(
        self,
        sumo_cfg: str,
        intersection_ids: List[str],
        use_gui: bool = False,
        max_steps: int = 3600,
        phase_duration: int = 30,
        yellow_duration: int = 3
    ):
        self.sumo_cfg = sumo_cfg
        self.intersection_ids = intersection_ids
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.phase_duration = phase_duration
        self.yellow_duration = yellow_duration
        
        # State management
        self.intersections = {}
        self.current_step = 0
        self.total_waiting_time = 0
        self.vehicle_count = 0
        
        # Performance metrics
        self.metrics = {
            'waiting_times': [],
            'queue_lengths': [],
            'throughput': [],
            'stops': []
        }
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial states"""
        # Close existing connection if any
        if traci.isLoaded():
            traci.close()
            
        # Start SUMO
        sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        sumo_cmd = [sumo_binary, '-c', self.sumo_cfg, '--no-warnings']
        traci.start(sumo_cmd)
        
        # Initialize intersections
        self.intersections = {
            int_id: IntersectionState(int_id) 
            for int_id in self.intersection_ids
        }
        
        # Reset counters
        self.current_step = 0
        self.total_waiting_time = 0
        self.vehicle_count = 0
        
        # Get initial states
        states = {}
        for int_id, intersection in self.intersections.items():
            intersection.update_from_sumo(int_id)
            states[int_id] = intersection.get_state_vector()
            
        return states
    
    def step(
        self, 
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """
        Execute actions and simulate for phase_duration steps.
        Returns: next_states, rewards, done, info
        """
        rewards = {}
        
        # Store previous waiting times for reward calculation
        prev_waiting_times = self._get_average_waiting_times()
        
        # Apply actions (set traffic light phases)
        for int_id, action in actions.items():
            self._set_traffic_phase(int_id, action)
            
        # Simulate for phase duration
        for _ in range(self.phase_duration):
            if self.current_step >= self.max_steps:
                break
                
            traci.simulationStep()
            self.current_step += 1
            
            # Collect metrics
            self._collect_metrics()
            
        # Get new states
        next_states = {}
        for int_id, intersection in self.intersections.items():
            intersection.update_from_sumo(int_id)
            next_states[int_id] = intersection.get_state_vector()
            
        # Calculate rewards (reduction in average waiting time)
        curr_waiting_times = self._get_average_waiting_times()
        for int_id in self.intersection_ids:
            rewards[int_id] = prev_waiting_times[int_id] - curr_waiting_times[int_id]
            
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'step': self.current_step,
            'avg_waiting_time': np.mean(list(curr_waiting_times.values())),
            'total_vehicles': len(traci.vehicle.getIDList())
        }
        
        return next_states, rewards, done, info
    
    def _set_traffic_phase(self, tls_id: str, phase: int):
        """Set traffic light phase with yellow transition if needed"""
        current_phase = traci.trafficlight.getPhase(tls_id)
        
        if current_phase != phase:
            # Insert yellow phase for safety
            yellow_state = self._get_yellow_state(tls_id)
            traci.trafficlight.setRedYellowGreenState(tls_id, yellow_state)
            
            for _ in range(self.yellow_duration):
                traci.simulationStep()
                self.current_step += 1
                
        # Set new phase
        phase_state = self._get_phase_state(tls_id, phase)
        traci.trafficlight.setRedYellowGreenState(tls_id, phase_state)
        traci.trafficlight.setPhase(tls_id, phase)
        
    def _get_phase_state(self, tls_id: str, phase: int) -> str:
        """
        Get traffic light state string for given phase.
        For Indian traffic: simplified phase control
        """
        # This is a simplified version - adapt based on your SUMO network
        phase_states = {
            0: "GGrrrrGGrrrr",  # N-S straight
            1: "rrGGrrrrGGrr",  # N-S left
            2: "rrrrGGrrrrGG",  # E-W straight  
            3: "rrrrrrGGrrGG",  # E-W left
        }
        return phase_states.get(phase, "rrrrrrrrrrrr")
    
    def _get_yellow_state(self, tls_id: str) -> str:
        """Get yellow state for transition"""
        return "yyyyyyyyyyyy"  # All yellow
    
    def _get_average_waiting_times(self) -> Dict[str, float]:
        """Get average waiting time per intersection"""
        waiting_times = {}
        
        for int_id in self.intersection_ids:
            total_wait = 0
            vehicle_count = 0
            
            controlled_lanes = traci.trafficlight.getControlledLanes(int_id)
            for lane in controlled_lanes:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                for vid in vehicle_ids:
                    total_wait += traci.vehicle.getWaitingTime(vid)
                    vehicle_count += 1
                    
            avg_wait = total_wait / max(vehicle_count, 1)
            waiting_times[int_id] = avg_wait
            
        return waiting_times
    
    def _collect_metrics(self):
        """Collect performance metrics"""
        all_vehicles = traci.vehicle.getIDList()
        
        if all_vehicles:
            # Waiting times
            wait_times = [traci.vehicle.getWaitingTime(vid) for vid in all_vehicles]
            self.metrics['waiting_times'].append(np.mean(wait_times))
            
            # Queue lengths
            total_queue = sum(
                traci.lane.getLastStepHaltingNumber(lane)
                for lane in traci.lane.getIDList()
            )
            self.metrics['queue_lengths'].append(total_queue)
            
            # Number of stops
            stops = [traci.vehicle.getAccumulatedWaitingTime(vid) for vid in all_vehicles]
            self.metrics['stops'].append(np.mean(stops))
            
    def close(self):
        """Close SUMO connection"""
        if traci.isLoaded():
            traci.close()
            
    def get_metrics(self) -> Dict:
        """Get collected performance metrics"""
        return {
            'avg_waiting_time': np.mean(self.metrics['waiting_times']) if self.metrics['waiting_times'] else 0,
            'avg_queue_length': np.mean(self.metrics['queue_lengths']) if self.metrics['queue_lengths'] else 0,
            'avg_stops': np.mean(self.metrics['stops']) if self.metrics['stops'] else 0,
            'total_throughput': len(self.metrics['waiting_times'])
        }