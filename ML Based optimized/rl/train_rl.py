import os
import sys
from pathlib import Path
import logging
import time
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils.data_loader import load_boxes_from_excel
from utils.synthetic_data import generate_synthetic_data, prepare_batch
from models.container import Container
from models.neural_net import PackingNetwork
from rl.packing_env import PackingEnvironment
from rl.dqn_agent import DQNAgent

def train_dqn(env, episodes=50,  # Reduced from 1000 to 50 episodes
              gamma=0.99, 
              lr=0.001,  # Slightly increased learning rate
              batch_size=64,  # Increased batch size
              update_target=5,  # More frequent target updates
              epsilon_start=1.0, 
              epsilon_end=0.01,
              epsilon_decay=0.90):  # Faster epsilon decay
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on device: {device}")
    logging.info("Quick training mode: 50 episodes")
    
    # Initialize agents with smaller network for faster training
    agent = DQNAgent(
        input_channels=1, 
        state_size=32, 
        action_dim=env.action_space
    ).to(device)
    
    target_agent = DQNAgent(
        input_channels=1,
        state_size=32,
        action_dim=env.action_space
    ).to(device)
    
    target_agent.load_state_dict(agent.state_dict())
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    
    # Training metrics
    best_reward = float('-inf')
    no_improvement_count = 0
    epsilon = epsilon_start
    
    for ep in range(episodes):
        start_time = time.time()
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        # Episode loop
        while True:
            # Select and perform action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = agent.select_action(obs_tensor, epsilon)
            next_obs, reward, done, _ = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Store transition and train
            if len(env.replay_buffer) >= batch_size:
                batch = env.sample_replay(batch_size)
                loss = train_step(agent, target_agent, optimizer, batch, gamma, device)
                
                if steps % 100 == 0:  # Log every 100 steps
                    logging.info(f"Step {steps}: Loss = {loss:.4f}, Reward = {total_reward:.2f}")
            
            if done:
                break
                
            obs = next_obs
        
        # Update target network more frequently
        if ep % update_target == 0:
            target_agent.load_state_dict(agent.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Log detailed episode info
        episode_time = time.time() - start_time
        logging.info(
            f"Episode {ep+1}/{episodes} - "
            f"Steps: {steps}, "
            f"Reward: {total_reward:.2f}, "
            f"Epsilon: {epsilon:.3f}, "
            f"Buffer Size: {len(env.replay_buffer)}, "
            f"Time: {episode_time:.2f}s"
        )
        
        # Early stopping check
        if total_reward > best_reward:
            best_reward = total_reward
            no_improvement_count = 0
            # Save best model
            torch.save(agent.state_dict(), 'best_model.pt')
        else:
            no_improvement_count += 1
        
        # Stop if no improvement for 20 episodes
        if no_improvement_count >= 20:
            logging.info("Early stopping: No improvement for 20 episodes")
            break

        # More frequent logging
        if (ep + 1) % 5 == 0:  # Log every 5 episodes instead of 10
            logging.info(
                f"Episode {ep+1}/{episodes} - "
                f"Reward: {total_reward:.2f}, "
                f"Steps: {steps}, "
                f"Epsilon: {epsilon:.3f}, "
                f"Buffer Size: {len(env.replay_buffer)}"
            )
        
        # Early stopping if good performance reached
        if total_reward > 50:  # Add early stopping condition
            logging.info(f"Good performance reached at episode {ep+1}")
            break
    
    return agent

def train_step(agent, target_agent, optimizer, batch, gamma, device):
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Convert lists to numpy arrays first for better performance
    states = np.array(states)
    next_states = np.array(next_states)
    
    # Convert numpy arrays to tensors
    states = torch.FloatTensor(states).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).unsqueeze(1).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.BoolTensor(dones).to(device)

    # Get current Q values
    current_q = agent(states)
    current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze()

    # Get next Q values with proper masking
    with torch.no_grad():
        next_q = target_agent(next_states).max(1)[0]
        # Create a masked version of next_q where done states are 0
        next_q_masked = next_q * (~dones).float()
        target_q = rewards + gamma * next_q_masked

    # Compute loss and update
    loss = torch.nn.functional.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

if __name__ == "__main__":
    # Setup environment with correct dimensions
    container = Container()
    boxes = load_boxes_from_excel('Larger_Box_Dimensions.xlsx')
    env = PackingEnvironment((32,32,32), boxes)
    
    print(f"Initialized environment with action space: {env.action_space}")
    trained_agent, metrics = train_dqn(env)
    
    # Save final model
    torch.save(trained_agent.state_dict(), 'final_model.pth')

    # Save metrics
    with open('training_metrics.json', 'w') as f:
        json.dump({
            'episode_rewards': metrics['rewards'],
            'successful_placements': metrics['successful_placements'],
            'losses': metrics['losses'],
            'training_time': metrics['training_time']
        }, f)