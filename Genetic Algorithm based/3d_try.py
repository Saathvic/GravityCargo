import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple
import math
from plotly.subplots import make_subplots

class ContainerItem:
    def __init__(self, name: str, length: float, width: float, height: float, 
                 weight: float, stackable: bool = True, max_stack_weight: float = None,
                 color: str = 'lightblue', rotation_allowed: bool = True):
        self.name = name
        self.dimensions = (length, width, height)
        self.weight = weight
        self.stackable = stackable
        self.max_stack_weight = max_stack_weight or weight * 3
        self.color = color
        self.rotation_allowed = rotation_allowed
        self.position = None
        self.rotation = 0  # 0, 90, 180, 270 degrees

class Container:
    def __init__(self, length: float, width: float, height: float, max_weight: float):
        self.length = length
        self.width = width
        self.height = height
        self.max_weight = max_weight
        self.items: List[ContainerItem] = []
        self.space_utilization = 0.0
        self.weight_utilization = 0.0
        self.loading_sequence = []
        self.weight_distribution = np.zeros((10, 10))  # Grid for weight distribution
        self.stability_score = 0.0
        self.optimization_suggestions = []

    def calculate_center_of_gravity(self) -> tuple:
        total_weight = sum(item.weight for item in self.items)
        if total_weight == 0:
            return (self.length/2, self.width/2, 0)
        
        x_cog = sum(item.position[0] * item.weight for item in self.items) / total_weight
        y_cog = sum(item.position[1] * item.weight for item in self.items) / total_weight
        z_cog = sum(item.position[2] * item.weight for item in self.items) / total_weight
        
        return (x_cog, y_cog, z_cog)

    def check_stability(self) -> bool:
        cog = self.calculate_center_of_gravity()
        # Check if COG is within base of support
        return (0 <= cog[0] <= self.length and 
                0 <= cog[1] <= self.width)

    def update_weight_distribution(self):
        """Calculate weight distribution across container floor"""
        grid_x = np.linspace(0, self.length, 10)
        grid_y = np.linspace(0, self.width, 10)
        self.weight_distribution = np.zeros((10, 10))
        
        for item in self.items:
            x, y, _ = item.position
            # Find grid position
            x_idx = int((x / self.length) * 9)
            y_idx = int((y / self.width) * 9)
            self.weight_distribution[y_idx, x_idx] += item.weight

    def check_collision(self, new_item: ContainerItem, position: Tuple[float, float, float]) -> bool:
        """Check if new item collides with existing items"""
        x, y, z = position
        for item in self.items:
            ix, iy, iz = item.position
            if (x < ix + item.dimensions[0] and x + new_item.dimensions[0] > ix and
                y < iy + item.dimensions[1] and y + new_item.dimensions[1] > iy and
                z < iz + item.dimensions[2] and z + new_item.dimensions[2] > iz):
                return True
        return False

    def optimize_placement(self):
        """Generate optimization suggestions"""
        self.optimization_suggestions = []
        
        # Check weight distribution
        if np.std(self.weight_distribution) > 1000:  # High variance in weight distribution
            self.optimization_suggestions.append("Consider redistributing weight more evenly")
        
        # Check vertical space utilization
        heights = [item.position[2] + item.dimensions[2] for item in self.items]
        if max(heights) < self.height * 0.7:
            self.optimization_suggestions.append("Vertical space underutilized - consider stacking")

def create_box_vertices(x, y, z, l, w, h):
    return {
        'x': [x, x + l, x + l, x, x, x + l, x + l, x],
        'y': [y, y, y + w, y + w, y, y, y + w, y + w],
        'z': [z, z, z, z, z + h, z + h, z + h, z + h]
    }

def create_box_faces(x, y, z, l, w, h):
    """Create faces for a 3D box with proper coloring"""
    # Define vertices for all faces
    vertices = np.array([
        # Front face
        [x, y, z], [x+l, y, z], [x+l, y, z+h], [x, y, z+h],
        # Back face
        [x, y+w, z], [x+l, y+w, z], [x+l, y+w, z+h], [x, y+w, z+h],
        # Left face
        [x, y, z], [x, y+w, z], [x, y+w, z+h], [x, y, z+h],
        # Right face
        [x+l, y, z], [x+l, y+w, z], [x+l, y+w, z+h], [x+l, y, z+h],
        # Bottom face
        [x, y, z], [x+l, y, z], [x+l, y+w, z], [x, y+w, z],
        # Top face
        [x, y, z+h], [x+l, y, z+h], [x+l, y+w, z+h], [x, y+w, z+h]
    ])
    
    # Create triangles for each face
    i = []
    j = []
    k = []
    
    # Add triangles for each face (2 triangles per face)
    for face in range(6):
        base = face * 4
        i.extend([base, base, base+2])
        j.extend([base+1, base+2, base+3])
        k.extend([base+1, base+3, base+3])
    
    return vertices[:, 0], vertices[:, 1], vertices[:, 2], np.array(i), np.array(j), np.array(k)

def create_3d_visualization(container: Container):
    # Create subplots for 3D view and weight distribution
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene', 'colspan': 2}, None],
               [{'type': 'heatmap'}, {'type': 'table'}]],
        subplot_titles=('Container Loading View', 'Weight Distribution', 'Loading Details')
    )

    # Add container outline with edges
    fig.add_trace(go.Mesh3d(
        x=[0, container.length, container.length, 0, 0, container.length, container.length, 0],
        y=[0, 0, container.width, container.width, 0, 0, container.width, container.width],
        z=[0, 0, 0, 0, container.height, container.height, container.height, container.height],
        opacity=0.1,
        color='lightgray',
        name="Container",
        showscale=False
    ), row=1, col=1)

    # Create a distinct color palette for better separation
    color_palette = [
        '#FF595E', '#FF924C', '#FFCA3A', '#8AC926',
        '#1982C4', '#6A4C93', '#FF6B6B', '#4ECDC4',
        '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5'
    ]

    # Store traces for hover highlighting
    box_traces = []
    edge_traces = []

    # Fix: Change the iteration to enumerate the items list
    for idx, item in enumerate(container.items):
        if item.position:
            x, y, z = item.position
            l, w, h = item.dimensions
            
            # Apply rotation if needed
            if item.rotation in [90, 270]:
                l, w = w, l

            # Create box faces with updated function
            x_coords, y_coords, z_coords, i, j, k = create_box_faces(x, y, z, l, w, h)
            
            # Assign color from palette
            item_color = color_palette[idx % len(color_palette)]
            
            # Add box with improved lighting and faces
            box_trace = go.Mesh3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                i=i, j=j, k=k,
                color=item_color,
                opacity=0.95,
                name=f"{item.name} ({item.weight}kg)",
                showscale=False,
                lighting=dict(
                    ambient=0.8,        # Increased ambient light
                    diffuse=1.0,        # Maximum diffuse light
                    fresnel=2.0,        # Enhanced fresnel for better edge visibility
                    specular=1.0,       # Maximum specular light
                    roughness=0.1       # Reduced roughness for smoother appearance
                ),
                lightposition=dict(
                    x=1000,
                    y=1000,
                    z=1000
                ),
                hovertemplate=
                f"<b>{item.name}</b><br>" +
                f"Weight: {item.weight}kg<br>" +
                f"Dimensions: {l:.1f}x{w:.1f}x{h:.1f}m<br>" +
                f"Position: ({x:.1f}, {y:.1f}, {z:.1f})<extra></extra>"
            )
            fig.add_trace(box_trace, row=1, col=1)
            box_traces.append(box_trace)

            # Add edges with slightly darker color for better definition
            darker_color = f'rgba({int(int(item_color[1:3], 16)*0.7)},{int(int(item_color[3:5], 16)*0.7)},{int(int(item_color[5:7], 16)*0.7)},1)'
            # Add edges with matching color
            edges_x, edges_y, edges_z = [], [], []
            for edge in [
                (0,1), (1,2), (2,3), (3,0),  # bottom
                (4,5), (5,6), (6,7), (7,4),  # top
                (0,4), (1,5), (2,6), (3,7)   # verticals
            ]:
                edges_x.extend([x_coords[edge[0]], x_coords[edge[1]], None])
                edges_y.extend([y_coords[edge[0]], y_coords[edge[1]], None])
                edges_z.extend([z_coords[edge[0]], z_coords[edge[1]], None])
            
            edge_trace = go.Scatter3d(
                x=edges_x, y=edges_y, z=edges_z,
                mode='lines',
                line=dict(color=darker_color, width=2),
                showlegend=False,
                hoverinfo='skip'
            )
            fig.add_trace(edge_trace, row=1, col=1)
            edge_traces.append(edge_trace)

    # Add COG indicator
    cog = container.calculate_center_of_gravity()
    fig.add_trace(go.Scatter3d(
        x=[cog[0]], y=[cog[1]], z=[cog[2]],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Center of Gravity'
    ), row=1, col=1)

    # Add weight distribution heatmap
    container.update_weight_distribution()
    fig.add_trace(
        go.Heatmap(
            z=container.weight_distribution,
            colorscale='Viridis',
            name='Weight Distribution (kg)',
            showscale=True
        ),
        row=2, col=1
    )

    # Add loading sequence table
    fig.add_trace(
        go.Table(
            header=dict(values=['Item', 'Weight (kg)', 'Position', 'Stability']),
            cells=dict(values=[
                [item.name for item in container.items],
                [item.weight for item in container.items],
                [f"({item.position[0]:.1f}, {item.position[1]:.1f}, {item.position[2]:.1f})" 
                 for item in container.items],
                ['✓' if item.position[2] == 0 or any(
                    other.position[2] + other.dimensions[2] == item.position[2]
                    for other in container.items if other != item
                ) else '!' for item in container.items]
            ])
        ),
        row=2, col=2
    )

    # Add optimization suggestions
    container.optimize_placement()
    if container.optimization_suggestions:
        fig.add_annotation(
            text='\n'.join(container.optimization_suggestions),
            xref="paper", yref="paper",
            x=0, y=-0.15,
            showarrow=False,
            font=dict(color="red")
        )

    # Update layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text=f"Container Loading Analysis<br>" +
                  f"Space: {container.space_utilization:.1f}% | " +
                  f"Weight: {container.weight_utilization:.1f}%"
    )

    return fig

# Example usage
container = Container(6.06, 2.44, 2.59, 28000)  # 20ft container with max weight in kg

# Create more realistic items with weights
items = [
    ContainerItem("Heavy Machinery", 2.0, 1.5, 1.2, 5000, stackable=False, color='red'),
    ContainerItem("Pallets", 1.2, 1.0, 1.4, 800, max_stack_weight=2400, color='orange'),
    ContainerItem("Boxes", 1.0, 1.0, 1.0, 500, color='blue'),
    ContainerItem("Fragile Equipment", 1.5, 1.2, 1.0, 1200, stackable=False, color='purple')
]

# Implement loading algorithm (simplified version)
def load_container(container: Container, items: List[ContainerItem]):
    """Enhanced loading algorithm with stability checks and collision detection"""
    current_x = 0
    current_y = 0
    current_z = 0
    layer_height = 0
    total_weight = 0
    
    # Sort items by weight (heavier items first) and stackability
    sorted_items = sorted(items, key=lambda x: (-x.weight, -x.stackable))
    
    for item in sorted_items:
        # Check weight constraint
        if total_weight + item.weight > container.max_weight:
            print(f"Warning: Cannot load {item.name} - weight limit exceeded")
            continue
            
        # Check if we need to start a new row
        if current_x + item.dimensions[0] > container.length:
            current_x = 0
            current_y += layer_height
            layer_height = 0
            
        # Check if we need to start a new layer
        if current_y + item.dimensions[1] > container.width:
            current_y = 0
            current_z += layer_height
            layer_height = 0
            
        # Check height constraint
        if current_z + item.dimensions[2] > container.height:
            print(f"Warning: Cannot load {item.name} - height limit exceeded")
            continue
            
        # Try different rotations if allowed
        best_position = None
        min_height = float('inf')
        
        if item.rotation_allowed:
            rotations = [0, 90]
        else:
            rotations = [0]
            
        for rotation in rotations:
            item.rotation = rotation
            l, w, h = item.dimensions if rotation == 0 else (item.dimensions[1], item.dimensions[0], item.dimensions[2])
            
            # Try different positions
            for x in np.arange(0, container.length - l + 0.1, 0.1):
                for y in np.arange(0, container.width - w + 0.1, 0.1):
                    # Find minimum height where item can be placed
                    z = 0
                    while True:
                        if not container.check_collision(item, (x, y, z)):
                            if z < min_height:
                                min_height = z
                                best_position = (x, y, z, rotation)
                            break
                        z += 0.1
                        if z > container.height:
                            break
        
        if best_position:
            x, y, z, rotation = best_position
            item.position = (x, y, z)
            item.rotation = rotation
            container.items.append(item)
            container.loading_sequence.append(item)
            total_weight += item.weight
        
        # Update positions
        current_x += item.dimensions[0]
        layer_height = max(layer_height, item.dimensions[2])
        
    # Calculate utilization
    total_volume = container.length * container.width * container.height
    used_volume = sum(item.dimensions[0] * item.dimensions[1] * item.dimensions[2] 
                     for item in container.items)
    container.space_utilization = (used_volume / total_volume) * 100
    container.weight_utilization = (total_weight / container.max_weight) * 100

# Create more realistic items with weights
items = [
    ContainerItem("Heavy Machine A", 2.0, 1.5, 1.2, 3000, stackable=False, color='red'),
    ContainerItem("Heavy Machine B", 1.8, 1.4, 1.1, 2800, stackable=False, color='darkred'),
    ContainerItem("Pallet 1", 1.2, 1.0, 1.4, 800, max_stack_weight=2400, color='orange'),
    ContainerItem("Pallet 2", 1.2, 1.0, 1.4, 750, max_stack_weight=2400, color='darkorange'),
    ContainerItem("Box Stack 1", 1.0, 1.0, 1.0, 500, color='blue'),
    ContainerItem("Box Stack 2", 1.0, 1.0, 1.0, 450, color='lightblue'),
    ContainerItem("Fragile Equipment", 1.5, 1.2, 1.0, 1200, stackable=False, color='purple'),
    ContainerItem("Light Machinery", 1.3, 1.1, 0.9, 900, color='green')
]

# Create and load container
container = Container(6.06, 2.44, 2.59, 28000)  # 20ft container
load_container(container, items)

# Create visualization
fig = create_3d_visualization(container)

# Add loading statistics
print(f"\nLoading Statistics:")
print(f"Space Utilization: {container.space_utilization:.1f}%")
print(f"Weight Utilization: {container.weight_utilization:.1f}%")
print(f"Items Loaded: {len(container.items)}/{len(items)}")

# Show the visualization
fig.show()
    