import torch
import torch.nn as nn
import torch.nn.functional as F

class BoxEncoder(nn.Module):
    def __init__(self, input_dim=13):  # Changed from 10 to 13 to match saved weights
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.layers(x)

class ContainerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x.view(x.size(0), -1)

class PackingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_encoder = BoxEncoder()
        self.container_encoder = ContainerEncoder()
        
        # Updated dimensions to match saved weights
        self.combined_encoder = nn.Sequential(
            nn.Linear(128 + 16384, 512),  # Changed input size and increased hidden size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),  # Increased output size
            nn.ReLU()
        )
        
        self.position_head = nn.Sequential(
            nn.Linear(256, 128),  # Updated dimensions
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        self.orientation_head = nn.Sequential(
            nn.Linear(256, 128),  # Updated dimensions
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
    def forward(self, box_features, container_state):
        box_embedding = self.box_encoder(box_features)
        container_embedding = self.container_encoder(container_state)
        
        combined = torch.cat([box_embedding, container_embedding], dim=1)
        combined = self.combined_encoder(combined)
        
        pos = torch.sigmoid(self.position_head(combined))
        orient = F.log_softmax(self.orientation_head(combined), dim=1)
        
        return pos, orient

    def load_legacy_weights(self, state_dict):
        """Handle loading of legacy model weights"""
        new_state_dict = {}
        
        # Map old keys to new keys
        key_mapping = {
            'box_encoder.0': 'box_encoder.layers.0',
            'box_encoder.1': 'box_encoder.layers.1',
            'box_encoder.4': 'box_encoder.layers.4',
            'container_encoder.0': 'container_encoder.layers.0',
            'container_encoder.3': 'container_encoder.layers.3',
            'combined_encoder.0': 'combined_encoder.0',
            'combined_encoder.3': 'combined_encoder.3',
            'position_head.0': 'position_head.0',
            'position_head.2': 'position_head.2',
            'orientation_head.0': 'orientation_head.0',
            'orientation_head.2': 'orientation_head.2'
        }
        
        for old_key, new_key in key_mapping.items():
            if old_key + '.weight' in state_dict:
                new_state_dict[new_key + '.weight'] = state_dict[old_key + '.weight']
            if old_key + '.bias' in state_dict:
                new_state_dict[new_key + '.bias'] = state_dict[old_key + '.bias']
            if old_key + '.running_mean' in state_dict:
                new_state_dict[new_key + '.running_mean'] = state_dict[old_key + '.running_mean']
            if old_key + '.running_var' in state_dict:
                new_state_dict[new_key + '.running_var'] = state_dict[old_key + '.running_var']
        
        # Load the mapped weights
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded legacy weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
