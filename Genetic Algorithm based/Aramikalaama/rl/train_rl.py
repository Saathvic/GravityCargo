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

def train_dqn(env, episodes=1000, gamma=0.99, lr=1e-3,
              batch_size=32, update_target=100, 
              epsilon_start=1.0, epsilon_end=0.01,
              epsilon_decay=0.995):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on device: {device}")
    logging.info(f"Initial epsilon: {epsilon_start}")
    logging.info(f"Batch size: {batch_size}")

    # Initialize monitoring metrics
    rewards_history = []
    losses_history = []
    epsilon = epsilon_start
    best_reward = float('-inf')
    
    # Create agents
    agent = DQNAgent(input_channels=1, state_size=32, action_dim=env.action_space).to(device)
    target_agent = DQNAgent(input_channels=1, state_size=32, action_dim=env.action_space).to(device)
    target_agent.load_state_dict(agent.state_dict())
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    
    start_time = time.time()
    
    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        losses = []
        steps = 0
        invalid_actions = 0

        # Debug first state
        if ep == 0:
            logging.info(f"Initial state shape: {obs.shape}")
            logging.info(f"Initial state range: [{obs.min()}, {obs.max()}]")
        
        while True:
            # Convert state and get action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = agent.select_action(obs_tensor, epsilon)
            
            # Take step and log results
            next_obs, reward, done, info = env.step(action)
            
            if reward < 0:
                invalid_actions += 1
            
            total_reward += reward
            steps += 1

            # Sample and train
            if len(env.replay_buffer) >= batch_size:
                batch = env.sample_replay(batch_size)
                loss = train_step(agent, target_agent, optimizer, batch, gamma, device)
                losses.append(loss)
                
                # Monitor for training issues
                if np.isnan(loss):
                    logging.error("NaN loss detected!")
                    return None, None
            
            if done:
                break
            
            obs = next_obs
        
        # Update target network
        if ep % update_target == 0:
            target_agent.load_state_dict(agent.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Log episode statistics
        avg_loss = np.mean(losses) if losses else 0
        rewards_history.append(total_reward)
        losses_history.append(avg_loss)
        
        # Detailed logging every few episodes
        if (ep + 1) % 10 == 0:
            logging.info(
                f"Episode {ep+1}/{episodes} - "
                f"Reward: {total_reward:.2f}, "
                f"Avg Loss: {avg_loss:.4f}, "
                f"Steps: {steps}, "
                f"Invalid Actions: {invalid_actions}, "
                f"Epsilon: {epsilon:.3f}, "
                f"Buffer Size: {len(env.replay_buffer)}"
            )
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({
                'episode': ep,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': total_reward,
            }, 'best_model.pt')
    
    training_time = time.time() - start_time
    logging.info(f"\nTraining completed in {training_time:.2f} seconds")
    logging.info(f"Best reward achieved: {best_reward:.2f}")
    logging.info(f"Final average reward: {np.mean(rewards_history[-100:]):.2f}")
    
    return agent, {
        'rewards': rewards_history,
        'losses': losses_history,
        'best_reward': best_reward,
        'training_time': training_time
    }

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