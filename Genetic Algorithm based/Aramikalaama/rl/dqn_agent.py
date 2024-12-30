import torch
import torch.nn as nn
import numpy as np

class DQNAgent(nn.Module):
    def __init__(self, input_channels=1, state_size=32, action_dim=10):
        super().__init__()
        
        # Proper 3D CNN architecture
        self.network = nn.Sequential(
            # Input: [batch, channels=1, depth, height, width]
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),  # Reduce spatial dimensions
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            nn.Flatten(),
            # Adjust linear layer input size based on pooling operations
            nn.Linear(64 * (state_size//4) * (state_size//4) * (state_size//4), 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        self.action_dim = action_dim

    def forward(self, x):
        # Ensure input is in correct shape [batch, channel, depth, height, width]
        if len(x.shape) == 4:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        return self.network(x)

    def select_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            # Ensure state is properly shaped
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if len(state.shape) == 3:
                state = state.unsqueeze(0).unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()