import numpy as np
from collections import deque
import random

class PackingEnvironment:
    def __init__(self, container_dims, boxes, max_replay_size=50000):  # Increased buffer size
        self.container_dims = np.array(container_dims)
        self.boxes = boxes
        self.replay_buffer = deque(maxlen=max_replay_size)
        
        # Define action space
        self.grid_size = 32  # Discretization of container space
        self.n_orientations = 6  # Number of possible orientations
        
        # Action space size = position space (32^3) * orientations (6)
        self.action_space = self.grid_size**3 * self.n_orientations
        
        # Initialize state as 3D grid
        self.state = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        self.reset()
        
        # Add episode limits
        self.max_steps = 1000  # Maximum steps per episode
        self.current_step = 0
    
    def decode_action(self, action):
        """Convert flat action index to (position, orientation)"""
        pos_idx = action // self.n_orientations
        orientation = action % self.n_orientations
        
        # Convert position index to 3D coordinates
        z = pos_idx % self.grid_size
        y = (pos_idx // self.grid_size) % self.grid_size
        x = pos_idx // (self.grid_size * self.grid_size)
        
        # Normalize positions to container dimensions
        pos = np.array([
            x * self.container_dims[0] / self.grid_size,
            y * self.container_dims[1] / self.grid_size,
            z * self.container_dims[2] / self.grid_size
        ])
        
        return pos, orientation
    
    def encode_action(self, position, orientation):
        """Convert (position, orientation) to flat action index"""
        # Discretize positions
        x = int(position[0] * self.grid_size / self.container_dims[0])
        y = int(position[1] * self.grid_size / self.container_dims[1])
        z = int(position[2] * self.grid_size / self.container_dims[2])
        
        # Convert to flat index
        pos_idx = x * self.grid_size * self.grid_size + y * self.grid_size + z
        return pos_idx * self.n_orientations + orientation

    def reset(self):
        # 3D binary occupancy grid (e.g., 32x32x32),
        # or scaled to container_dims in cm
        self.state = np.zeros((32, 32, 32), dtype=np.float32)
        self.packed_boxes = []
        self.remaining_boxes = self.boxes.copy()
        self.current_step = 0
        return self._get_observation()

    def store_experience(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def sample_replay(self, batch_size):
        """Sample a batch of experiences from the replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return None
        return random.sample(self.replay_buffer, batch_size)
    
    def step(self, action):
        self.current_step += 1
        
        # Get current state
        current_state = self._get_observation()
        
        # Apply action and get reward
        position, orientation = self.decode_action(action)
        reward = self._apply_action(position, orientation)
        
        # Get next state
        next_state = self._get_observation()
        
        # Check termination conditions
        done = (len(self.remaining_boxes) == 0 or 
                self.current_step >= self.max_steps)
        
        # Add step penalty to encourage faster solutions
        reward -= 0.01  # Small penalty for each step
        
        # Store experience
        self.store_experience(current_state, action, reward, next_state, done)
        
        return next_state, reward, done, {}

    def _get_observation(self):
        # Return state + features of next box, or a combined encoding
        return self.state

    def _calculate_reward(self):
        # Example: reward for valid placement, penalty for overlap/out-of-bounds
        return 0.0

    def _apply_action(self, position, orientation):
        """Apply the selected action and return reward"""
        if not self.remaining_boxes:
            return -1.0  # Penalty for invalid action when no boxes left
            
        current_box = self.remaining_boxes[0]
        rotated_dims = self._get_rotated_dimensions(current_box.dimensions, orientation)
        
        # Convert position to grid coordinates
        grid_pos = self._normalize_position(position)
        grid_dims = self._normalize_position(rotated_dims)
        
        # Ensure dimensions are at least 1 grid cell
        grid_dims = np.maximum(grid_dims, 1)
        
        # Check if placement is valid
        if self._is_valid_placement(grid_pos, grid_dims):
            # Update state
            self._update_state(grid_pos, grid_dims)
            self.packed_boxes.append({
                'box': current_box,
                'position': position,
                'dimensions': rotated_dims
            })
            self.remaining_boxes.pop(0)
            
            # Calculate reward based on placement quality
            reward = self._calculate_placement_reward(grid_pos, grid_dims)
            return reward
        else:
            return -1.0  # Penalty for invalid placement
    
    def _normalize_position(self, position):
        """Convert real coordinates to grid coordinates"""
        norm_pos = np.array(position) / self.container_dims * self.grid_size
        return np.clip(norm_pos, 0, self.grid_size-1).astype(int)
    
    def _denormalize_position(self, grid_pos):
        """Convert grid coordinates to real coordinates"""
        return np.array(grid_pos) * self.container_dims / self.grid_size
    
    def _get_rotated_dimensions(self, dims, orientation):
        """Get box dimensions after rotation"""
        l, w, h = dims
        orientations = [
            (l, w, h), (l, h, w),
            (w, l, h), (w, h, l),
            (h, l, w), (h, w, l)
        ]
        return orientations[orientation % 6]
    
    def _is_valid_placement(self, position, dimensions):
        """Check if placement is valid"""
        x, y, z = position
        l, w, h = dimensions
        
        # Check boundaries
        if (x < 0 or y < 0 or z < 0 or
            x + l > self.grid_size or
            y + w > self.grid_size or
            z + h > self.grid_size):
            return False
            
        # Check overlap with existing boxes
        x1, x2 = x, x + l
        y1, y2 = y, y + w
        z1, z2 = z, z + h
        
        if np.any(self.state[x1:x2, y1:y2, z1:z2] > 0):
            return False
            
        return True
    
    def _update_state(self, position, dimensions):
        """Update container state after placing a box"""
        x, y, z = position
        l, w, h = dimensions
        
        # Convert to grid coordinates
        x1, x2 = x, x + l
        y1, y2 = y, y + w
        z1, z2 = z, z + h
        
        # Update occupancy grid
        self.state[x1:x2, y1:y2, z1:z2] = 1
    
    def _calculate_placement_reward(self, grid_pos, grid_dims):
        # Increase rewards for good placements
        reward = 10.0  # Base reward for successful placement
        
        # Add larger bonuses for efficient placement
        volume_utilization = np.prod(grid_dims) / (self.grid_size ** 3)
        reward += volume_utilization * 5.0
        
        # Check adjacent cells in grid coordinates
        x, y, z = grid_pos
        l, w, h = grid_dims
        
        adjacent_filled = 0
        directions = [
            (-1,0,0), (1,0,0),  # Left, Right
            (0,-1,0), (0,1,0),  # Front, Back
            (0,0,-1), (0,0,1)   # Bottom, Top
        ]
        
        for dx, dy, dz in directions:
            check_x = x + dx * l
            check_y = y + dy * w
            check_z = z + dz * h
            
            if (0 <= check_x < self.grid_size and
                0 <= check_y < self.grid_size and
                0 <= check_z < self.grid_size):
                if np.any(self.state[check_x, check_y, check_z] > 0):
                    adjacent_filled += 1
        
        # Add bonus for each adjacent filled space
        reward += 0.1 * adjacent_filled
        
        # Add bonus for utilizing bottom space first (using grid coordinates)
        reward += (1.0 - y/self.grid_size) * 0.5
        
        return reward