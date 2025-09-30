1. To Train a New Model
This command starts the training process using Federated Learning, which is the default. It will run for 200 episodes and save model checkpoints and logs in a new ./checkpoints directory.

```bash
python main.py --mode train
```

To train using a different strategy:

```bash
# For a single agent (non-federated)
python main.py --mode train --training-mode individual
```
```bash
# For centralized training (all data sent to one agent)
python main.py --mode train --training-mode aggregated
```

2. To Test Your Trained Model
Once training is complete, you will have model files in your ./checkpoints folder. Use this command to run a simulation with your smart agent.

```bash
# Replace the file name with the actual checkpoint you want to test
python main.py --mode test --model-path ./checkpoints/federated_checkpoint_ep200.pt
```

3. To Run the Baseline Fixed-Time Controller
This runs the simulation with a simple, non-AI controller that switches lights on a fixed timer. This is useful for establishing a performance baseline.

```bash
python main.py --mode fixed
```

4. To Directly Compare Your Model vs. the Baseline
This command runs both the smart agent and the fixed-time agent and then prints a summary of how much better your FDRL model performed in terms of waiting time, queue length, etc.

```bash
# Replace the file name with your trained model
python main.py --mode compare --model-path ./checkpoints/federated_checkpoint_ep200.pt
```