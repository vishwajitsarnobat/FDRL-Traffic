import socket
import pickle
import torch
import numpy as np
from ppo_agent import PPOAgent, Memory
from sumo_simulator import SumoSimulator

class FederatedClient:
    def __init__(self, junction_info, config):
        self.junction_id = junction_info['id']
        self.incoming_roads = junction_info['incoming_roads']
        self.action_dim = len(self.incoming_roads)
        self.state_dim = 2 * self.action_dim
        self.config = config
        
        self.agent = PPOAgent(self.state_dim, self.action_dim, config)
        self.memory = Memory()
        
        self.server_host = config['system']['server_host']
        self.server_port = config['system']['server_port']
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect_to_server(self):
        self.socket.connect((self.server_host, self.server_port))
        print(f"Client {self.junction_id} connected to server.")
        
        # Send metadata to server for initialization
        meta_data = {
            'junction_id': self.junction_id,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        self.socket.sendall(pickle.dumps(meta_data))

    def run(self):
        self.connect_to_server()
        
        # Each client needs its own SUMO instance
        sim = SumoSimulator(
            self.config['sumo']['config_file'],
            step_length=self.config['sumo']['step_length'],
            gui=False # No GUI during training
        )
        
        for epoch in range(self.config['fdrl']['epochs']):
            # 1. Receive and load global model
            data_size = int.from_bytes(self.socket.recv(8), 'big')
            received_data = b""
            while len(received_data) < data_size:
                received_data += self.socket.recv(4096)
            global_weights = pickle.loads(received_data)
            
            self.agent.actor.load_state_dict(global_weights)
            self.agent.actor_old.load_state_dict(global_weights)
            
            print(f"Client {self.junction_id} received global model for epoch {epoch+1}.")

            # 2. Perform K local training steps
            cumulative_reward = 0
            
            for k_step in range(self.config['fdrl']['K']):
                state = sim.get_state(self.junction_id)
                action, log_prob = self.agent.select_action(state)
                
                # Store experience
                self.memory.states.append(state)
                self.memory.actions.append(action)
                self.memory.logprobs.append(log_prob)
                
                # Execute action in SUMO
                sim.set_phase(
                    self.junction_id, action,
                    self.config['fdrl']['yellow_time'],
                    self.config['fdrl']['green_time']
                )
                
                reward = sim.get_reward(self.junction_id)
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(False) # Assuming non-terminal environment
                cumulative_reward += reward

            # Update policy
            loss, actor_loss, critic_loss = self.agent.update(self.memory)
            self.memory.clear_memory()
            
            # 3. Send updated local model and logs to server
            log_data = {
                'cumulative_reward': cumulative_reward,
                'actor_loss': actor_loss,
                'critic_loss': critic_loss
            }
            payload = {
                'weights': self.agent.actor.state_dict(),
                'log': log_data
            }
            data_bytes = pickle.dumps(payload)
            self.socket.sendall(len(data_bytes).to_bytes(8, 'big'))
            self.socket.sendall(data_bytes)
            
            print(f"Client {self.junction_id} sent local model for epoch {epoch+1}.")

        sim.close()
        self.socket.close()