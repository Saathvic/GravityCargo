import pandas as pd
from models.box import Box
import torch
from torch.utils.data import Dataset
import numpy as np

def load_boxes_from_excel(file_path):
    df = pd.read_excel(file_path)
    boxes = []
    
    for _, row in df.iterrows():
        try:
            # Create box instances based on quantity
            quantity = int(row['Quantity'])
            for i in range(quantity):
                box = Box(
                    box_id=f"{row['Box ID']}_{i+1}",
                    category=row['Box Category'],
                    dimensions=row['Dimensions (cm)'],
                    weight=float(row['Weight (kg)']),
                    fragility=row['Fragility'],
                    load_bearing=float(row['Load Bearing (kg)']),
                    quantity=1,  # Individual box
                    shape=row['Shape of Packaging'],
                    packaging_type=row['Packaging Type'],
                    temp_req=row['Temperature Recommended'],
                    contents=row['Products Packed in Box']
                )
                boxes.append(box)
        except Exception as e:
            print(f"Error processing row: {row['Box ID']}, Error: {str(e)}")
            continue
    
    return boxes

class PackingDataset(Dataset):
    def __init__(self, boxes, container, placements):
        self.boxes = boxes
        self.container = container
        self.placements = placements
        
    def __len__(self):
        return len(self.placements)
    
    def __getitem__(self, idx):
        placement = self.placements[idx]
        box = placement['box']
        
        # Get box features
        box_features = torch.tensor(box.get_feature_vector(), dtype=torch.float32)
        
        # Get container state
        container_state = torch.tensor(self.container.space_matrix[None, :], dtype=torch.float32)
        
        # Get target position (normalized)
        target_position = torch.tensor(
            placement['position'] / self.container.dimensions,
            dtype=torch.float32
        )
        
        # Get target orientation (one-hot encoded)
        target_orientation = torch.tensor(placement['orientation'], dtype=torch.long)
        
        return box_features, container_state, target_position, target_orientation
