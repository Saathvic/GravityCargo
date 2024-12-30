import numpy as np
import torch

def generate_synthetic_data(boxes, container, num_samples=1000):
    # Normalize all boxes first
    for box in boxes:
        box.normalize_features(container.dimensions)
    
    training_data = []
    
    for _ in range(num_samples):
        # Randomly select a box
        box = np.random.choice(boxes)
        
        # Generate valid position
        pos_x = np.random.uniform(0, container.length - box.dimensions[0])
        pos_y = np.random.uniform(0, container.width - box.dimensions[1])
        pos_z = np.random.uniform(0, container.height - box.dimensions[2])
        
        # Generate random orientation (0-5 for 6 possible orientations)
        orientation = np.random.randint(0, 6)
        
        # Create container state matrix (32x32x32)
        container_state = np.zeros((32, 32, 32))
        
        # Add current box to container state (simplified)
        x, y, z = int(32 * pos_x/container.length), int(32 * pos_y/container.width), int(32 * pos_z/container.height)
        container_state[max(0, min(x, 31)):max(0, min(x+2, 32)), 
                       max(0, min(y, 31)):max(0, min(y+2, 32)), 
                       max(0, min(z, 31)):max(0, min(z+2, 32))] = 1
        
        training_data.append({
            'box_features': box.get_feature_vector(),
            'container_state': container_state.astype(np.float32),  # Ensure float32
            'position': np.array([pos_x, pos_y, pos_z]) / container.dimensions,  # Normalize positions
            'orientation': orientation
        })
    
    return training_data

def prepare_batch(training_data):
    box_features = torch.FloatTensor([d['box_features'] for d in training_data])
    # Reshape container states correctly for conv3d
    container_states = torch.FloatTensor([d['container_state'] for d in training_data])
    container_states = container_states.unsqueeze(1)  # Add channel dimension [batch, channel, depth, height, width]
    positions = torch.FloatTensor([d['position'] for d in training_data])
    orientations = torch.LongTensor([d['orientation'] for d in training_data])
    
    return box_features, container_states, positions, orientations
