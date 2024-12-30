import torch
from torch.utils.data import DataLoader, TensorDataset
from models.neural_net import PackingNetwork
from utils.synthetic_data import generate_synthetic_data, prepare_batch
from models.container import Container
from utils.data_loader import load_boxes_from_excel

def train():
    # Initialize components
    container = Container()
    boxes = load_boxes_from_excel('Larger_Box_Dimensions.xlsx')
    model = PackingNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Generate and prepare training data
    print("Generating synthetic training data...")
    training_data = generate_synthetic_data(boxes, container)
    box_features, container_states, positions, orientations = prepare_batch(training_data)
    
    # Create dataset and dataloader
    dataset = TensorDataset(box_features, container_states, positions, orientations)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    print("Starting training...")
    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_features, batch_states, batch_positions, batch_orientations in train_loader:
            optimizer.zero_grad()
            
            # Forward pass (no need to reshape batch_states anymore)
            pred_positions, pred_orientations = model(batch_features, batch_states)
            
            # Calculate loss
            position_loss = torch.nn.MSELoss()(pred_positions, batch_positions)
            orientation_loss = torch.nn.CrossEntropyLoss()(pred_orientations, batch_orientations)
            loss = position_loss + orientation_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Training completed!")

if __name__ == "__main__":
    train()
