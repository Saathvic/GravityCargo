import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import traceback
from typing import List, Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass
import random
import pandas as pd

class Fragility(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class ShapeType(Enum):
    REGULAR = 1
    IRREGULAR = 2

@dataclass
class ProductSpecs:
    name: str
    dimensions: Tuple[float, float, float]
    weight: float
    quantity: int
    fragility: Fragility
    stackable: bool
    shape_type: ShapeType
    boxing_type: str
    environment: Dict[str, float]
    priority: int

class PackedItem:
    def __init__(self, product: ProductSpecs, position: Tuple[float, float, float], rotation: int = 0):
        self.product = product
        self.position = position
        self.rotation = rotation  # 0, 90, 180, 270
        self.color = self._assign_color()

    def _assign_color(self) -> str:
        # Assign colors based on fragility and priority
        if self.product.fragility == Fragility.HIGH:
            return 'rgba(255, 0, 0, 0.7)'  # Red for high fragility
        elif self.product.priority == 1:
            return 'rgba(0, 255, 0, 0.7)'  # Green for high priority
        else:
            return f'rgba({random.randint(100,255)}, {random.randint(100,255)}, {random.randint(100,255)}, 0.7)'

class Layer:
    def __init__(self, height: float, items: List[PackedItem] = None):
        self.height = height
        self.items = items or []
        self.weight = sum(item.product.weight for item in self.items)

class SmartContainer:
    def __init__(self, length: float, width: float, height: float, max_weight: float):
        self.dimensions = (length, width, height)
        self.max_weight = max_weight
        self.packed_items: List[PackedItem] = []
        self.space_matrix = np.zeros((int(length*10), int(width*10), int(height*10)), dtype=bool)
        self.current_weight = 0
        self.volume_utilized = 0
        self.layers: List[Layer] = []
        self.environmental_zones = {
            'ambient': [(0, length), (0, width), (0, height)],
            'chilled': [(0, length/2), (0, width), (0, height/3)],
            # Add more zones as needed
        }
        self.skyline = [(0, 0, 0, length)]  # x, y, z, width
        self.cushioning_blocks = []  # Store cushioning blocks
        self.failed_section = {
            'start_x': length + 1,
            'current_x': length + 1,
            'current_y': 0,
            'current_z': 0,
            'row_height': 0
        }

    def add_cushioning(self, item: PackedItem, thickness: float = 0.05):
        """Add cushioning around fragile items"""
        if item.product.fragility == Fragility.HIGH:
            x, y, z = item.position
            l, w, h = item.product.dimensions
            
            # Create cushioning blocks
            cushion_positions = [
                # Bottom
                (x, y, z-thickness, l, w, thickness),
                # Top
                (x, y, z+h, l, w, thickness),
                # Sides
                (x-thickness, y, z, thickness, w, h),
                (x+l, y, z, thickness, w, h),
                (x, y-thickness, z, l, thickness, h),
                (x, y+w, z, l, thickness, h)
            ]
            
            for pos in cushion_positions:
                self.cushioning_blocks.append({
                    'position': pos[:3],
                    'dimensions': pos[3:],
                    'color': 'rgba(200, 200, 200, 0.3)'
                })

    def update_skyline(self, item: PackedItem):
        """Update skyline after placing an item"""
        x, y, z = item.position
        l, w, h = item.product.dimensions
        if item.rotation in [90, 270]:
            l, w = w, l
        
        new_height = z + h
        affected_segments = []
        new_segments = []
        
        for i, (sx, sy, sz, sw) in enumerate(self.skyline):
            if x < sx + sw and x + l > sx:
                affected_segments.append(i)
                
                # Create new segments
                if x > sx:
                    new_segments.append((sx, sy, sz, x - sx))
                if x + l < sx + sw:
                    new_segments.append((x + l, sy, sz, (sx + sw) - (x + l)))
                    
                new_segments.append((max(sx, x), y + w, new_height, 
                                  min(sx + sw, x + l) - max(sx, x)))
        
        # Remove affected segments and add new ones
        self.skyline = [s for i, s in enumerate(self.skyline) 
                       if i not in affected_segments] + new_segments
        self.skyline.sort(key=lambda x: x[0])  # Sort by x coordinate

    def find_best_position_skyline(self, product: ProductSpecs) -> Optional[Tuple[float, float, float, int]]:
        """Find best position using skyline placement algorithm"""
        best_height = float('inf')
        best_pos = None
        best_rotation = 0
        
        rotations = [0, 90, 180, 270] if product.shape_type == ShapeType.REGULAR else [0]
        
        for rotation in rotations:
            l, w, h = product.dimensions
            if rotation in [90, 270]:
                l, w = w, l
                
            # Try each skyline segment
            for i, (sx, sy, sz, sw) in enumerate(self.skyline):
                if sw >= l:  # Segment wide enough
                    # Check if item fits in container
                    if (sx + l <= self.dimensions[0] and 
                        sy + w <= self.dimensions[1] and 
                        sz + h <= self.dimensions[2]):
                        
                        # Create temporary item to check constraints
                        temp_item = PackedItem(product, (sx, sy, sz), rotation)
                        
                        if (not self._check_collision((sx, sy, sz), (l, w, h)) and 
                            self.is_stable(temp_item) and 
                            validate_placement(self, temp_item)):
                            
                            if sz + h < best_height:
                                best_height = sz + h
                                best_pos = (sx, sy, sz)
                                best_rotation = rotation
        
        return (best_pos[0], best_pos[1], best_pos[2], best_rotation) if best_pos else None

    def check_constraints(self, item: ProductSpecs, position: Tuple[float, float, float], 
                         rotation: int) -> bool:
        x, y, z = position
        l, w, h = item.dimensions
        
        # Adjust dimensions based on rotation
        if rotation in [90, 270]:
            l, w = w, l

        # Check boundaries
        if (x + l > self.dimensions[0] or y + w > self.dimensions[1] or 
            z + h > self.dimensions[2]):
            return False

        # Check weight constraints
        if self.current_weight + item.weight > self.max_weight:
            return False

        # Check collision with existing items
        return not self._check_collision(position, (l, w, h))

    def _check_collision(self, position: Tuple[float, float, float], 
                        dimensions: Tuple[float, float, float]) -> bool:
        x, y, z = [int(p * 10) for p in position]
        l, w, h = [int(d * 10) for d in dimensions]
        
        return np.any(self.space_matrix[x:x+l, y:y+w, z:z+h])

    def _update_space_matrix(self, item: PackedItem):
        x, y, z = [int(p * 10) for p in item.position]
        l, w, h = [int(d * 10) for d in item.product.dimensions]
        if item.rotation in [90, 270]:
            l, w = w, l
        
        self.space_matrix[x:x+l, y:y+w, z:z+h] = True

    def is_stable(self, item: PackedItem) -> bool:
        """Check if item placement is stable"""
        if item.position[2] == 0:  # On floor
            return True
            
        support_area = 0
        item_base_area = item.product.dimensions[0] * item.product.dimensions[1]
        
        for other in self.packed_items:
            if abs(other.position[2] + other.product.dimensions[2] - item.position[2]) < 0.01:
                # Calculate overlapping area
                overlap = self._calculate_overlap(item, other)
                support_area += overlap
                
        return support_area >= 0.7 * item_base_area  # 70% support requirement

    def _calculate_overlap(self, item1: PackedItem, item2: PackedItem) -> float:
        """Calculate overlapping area between two items"""
        x1, y1, _ = item1.position
        x2, y2, _ = item2.position
        
        dx = min(x1 + item1.product.dimensions[0], x2 + item2.product.dimensions[0]) - \
             max(x1, x2)
        dy = min(y1 + item1.product.dimensions[1], y2 + item2.product.dimensions[1]) - \
             max(y1, y2)
             
        if dx > 0 and dy > 0:
            return dx * dy
        return 0

def load_products_from_csv(filepath: str) -> List[ProductSpecs]:
    """Load products from CSV file"""
    df = pd.read_csv(filepath)
    products = []
    
    for _, row in df.iterrows():
        products.append(ProductSpecs(
            name=row['Name'],
            dimensions=(float(row['Length']), float(row['Width']), float(row['Height'])),
            weight=float(row['Weight']),
            quantity=int(row['Quantity']),
            fragility=Fragility[row['Fragility'].upper()],
            stackable=row['Stackable'].lower() == 'yes',
            shape_type=ShapeType[row['ShapeType'].upper()],
            boxing_type=row['BoxingType'],
            environment={'temp': float(row['Temperature'])},
            priority=int(row['Priority'])
        ))
    
    return products

def find_stable_height(container: SmartContainer, product: ProductSpecs, 
                      position: Tuple[float, float, float], rotation: int) -> Optional[float]:
    """Find stable height for item placement"""
    x, y, _ = position
    l, w, h = product.dimensions
    if rotation in [90, 270]:
        l, w = w, l
    
    max_height = 0
    support_found = False
    
    # Check support from items below
    for item in container.packed_items:
        ix, iy, iz = item.position
        il, iw, ih = item.product.dimensions
        
        # Check if there's overlap in xy plane
        if (x < ix + il and x + l > ix and
            y < iy + iw and y + w > iy):
            support_found = True
            max_height = max(max_height, iz + ih)
    
    return max_height if support_found or max_height == 0 else None

def pack_items(container: SmartContainer, products: List[ProductSpecs]) -> Tuple[bool, List[str], List[Tuple[ProductSpecs, str]]]:
    """Enhanced packing algorithm using skyline placement with better error handling"""
    errors = []
    failed_items = []
    
    # Pre-check validation
    total_volume = sum(p.dimensions[0] * p.dimensions[1] * p.dimensions[2] * p.quantity 
                      for p in products)
    container_volume = container.dimensions[0] * container.dimensions[1] * container.dimensions[2]
    
    print(f"\nVolume Check:")
    print(f"- Total product volume: {total_volume:.2f}m³")
    print(f"- Container volume: {container_volume:.2f}m³")
    print(f"- Required capacity: {(total_volume/container_volume)*100:.1f}%")
    
    # Sort by priority and size
    sorted_products = sorted(products, key=lambda p: (
        -p.priority,  # Higher priority first
        -p.dimensions[0] * p.dimensions[1] * p.dimensions[2],  # Larger volume first
        -p.weight,  # Heavier items first
        p.fragility.value  # Less fragile first
    ))

    # Pack items by layers
    current_layer_height = 0
    max_layer_height = container.dimensions[2]
    
    for product in sorted_products:
        print(f"\nTrying to pack: {product.name}")
        for i in range(product.quantity):
            # Try to place in current layer first
            position = None
            best_rotation = 0
            min_wasted_space = float('inf')
            
            # Try all possible rotations
            for rotation in ([0, 90, 180, 270] if product.shape_type == ShapeType.REGULAR else [0]):
                l, w, h = product.dimensions
                if rotation in [90, 270]:
                    l, w = w, l
                    
                # Try positions in current layer
                for x in np.arange(0, container.dimensions[0] - l + 0.1, 0.1):
                    for y in np.arange(0, container.dimensions[1] - w + 0.1, 0.1):
                        # Check if position is valid
                        if container.check_constraints(product, (x, y, current_layer_height), rotation):
                            # Calculate wasted space
                            wasted_space = calculate_wasted_space(container, (x, y, current_layer_height), (l, w, h))
                            if wasted_space < min_wasted_space:
                                min_wasted_space = wasted_space
                                position = (x, y, current_layer_height)
                                best_rotation = rotation
            
            if position is None and current_layer_height + product.dimensions[2] <= max_layer_height:
                # Try starting a new layer
                current_layer_height = get_next_layer_height(container)
                continue
                
            if position:
                # Create and validate packed item
                packed_item = PackedItem(product, position, best_rotation)
                if validate_placement_with_details(container, packed_item)[0]:
                    container.packed_items.append(packed_item)
                    container._update_space_matrix(packed_item)
                    print(f"- Successfully packed at position {position}")
                else:
                    failed_items.append((product, "Failed validation checks"))
                    print(f"- Failed validation at position {position}")
            else:
                failed_items.append((product, "No valid position found"))
                print(f"- No valid position found")

    success = len(failed_items) == 0
    return success, errors, failed_items

def calculate_wasted_space(container: SmartContainer, position: Tuple[float, float, float], 
                         dimensions: Tuple[float, float, float]) -> float:
    """Calculate wasted space for a given position"""
    x, y, z = position
    l, w, h = dimensions
    wasted_space = 0
    
    # Check gaps with neighboring items
    for item in container.packed_items:
        if abs(item.position[2] - z) < 0.01:  # Same layer
            dx = min(x + l, item.position[0] + item.product.dimensions[0]) - \
                 max(x, item.position[0])
            dy = min(y + w, item.position[1] + item.product.dimensions[1]) - \
                 max(y, item.position[1])
            if dx > 0 and dy > 0:
                wasted_space += dx * dy * h
                
    return wasted_space

def get_next_layer_height(container: SmartContainer) -> float:
    """Find the height for the next layer"""
    if not container.packed_items:
        return 0
    
    current_heights = [item.position[2] + item.product.dimensions[2] 
                      for item in container.packed_items]
    return max(current_heights)

def sort_products_by_constraints(products: List[ProductSpecs]) -> List[ProductSpecs]:
    """Sort products considering multiple constraints"""
    return sorted(products, key=lambda p: (
        -p.priority,  # Higher priority first
        -p.weight,    # Heavier items first
        -p.fragility.value,  # More fragile items later
        -(p.dimensions[0] * p.dimensions[1] * p.dimensions[2])  # Larger volume first
    ))

def validate_placement(container: SmartContainer, item: PackedItem) -> bool:
    """Validate item placement against all constraints"""
    # Check weight distribution
    if not check_weight_distribution(container, item):
        return False
    
    # Check environmental constraints
    if not check_environmental_constraints(container, item):
        return False
    
    # Check fragility constraints
    if not check_fragility_constraints(container, item):
        return False
    
    # Check stability
    if not container.is_stable(item):
        return False
    
    return True

def validate_placement_with_details(container: SmartContainer, item: PackedItem) -> Tuple[bool, str]:
    """Detailed placement validation with specific error messages"""
    
    # Weight distribution check
    if not check_weight_distribution(container, item):
        return False, "Would cause uneven weight distribution"

    # Environmental constraints check
    if not check_environmental_constraints(container, item):
        return False, f"Environmental requirements not met (temp: {item.product.environment['temp']}°C)"

    # Fragility constraints check
    if not check_fragility_constraints(container, item):
        return False, "Fragile item placement violated"

    # Stability check
    if not container.is_stable(item):
        return False, "Item would be unstable at this position"

    # Check stacking constraints
    if not check_stacking_constraints(container, item):
        return False, "Stacking constraints violated"

    return True, ""

# Add helper functions for constraint checking
def check_weight_distribution(container: SmartContainer, item: PackedItem) -> bool:
    """Check if adding the item maintains good weight distribution"""
    # Create a temporary weight distribution grid
    grid_size = 10
    weight_grid = np.zeros((grid_size, grid_size))
    
    # Map existing items to grid
    for existing_item in container.packed_items:
        x, y, _ = existing_item.position
        grid_x = int((x / container.dimensions[0]) * (grid_size - 1))
        grid_y = int((y / container.dimensions[1]) * (grid_size - 1))
        weight_grid[grid_x, grid_y] += existing_item.product.weight
    
    # Add new item
    x, y, _ = item.position
    grid_x = int((x / container.dimensions[0]) * (grid_size - 1))
    grid_y = int((y / container.dimensions[1]) * (grid_size - 1))
    weight_grid[grid_x, grid_y] += item.product.weight
    
    # Check weight distribution
    max_weight_diff = container.max_weight * 0.3  # Allow 30% difference
    return np.max(weight_grid) - np.min(weight_grid) < max_weight_diff

def check_environmental_constraints(container: SmartContainer, item: PackedItem) -> bool:
    """Check if item's environmental requirements are met"""
    x, y, z = item.position
    temp_requirement = item.product.environment.get('temp', 20)  # Default temp 20°C
    
    # Check if item is in appropriate temperature zone
    if temp_requirement <= 18:  # Cold items
        zone = container.environmental_zones['chilled']
        return (x <= zone[0][1] and y <= zone[1][1] and z <= zone[2][1])
    return True

def check_fragility_constraints(container: SmartContainer, item: PackedItem) -> bool:
    """Check if fragile items are properly handled"""
    if item.product.fragility == Fragility.HIGH:
        # Ensure no items are placed on top of fragile items
        for existing_item in container.packed_items:
            if (existing_item.position[0] < item.position[0] + item.product.dimensions[0] and
                existing_item.position[0] + existing_item.product.dimensions[0] > item.position[0] and
                existing_item.position[1] < item.position[1] + item.product.dimensions[1] and
                existing_item.position[1] + existing_item.product.dimensions[1] > item.position[1] and
                existing_item.position[2] > item.position[2]):
                return False
    return True

def check_stacking_constraints(container: SmartContainer, item: PackedItem) -> bool:
    """Check if stacking rules are followed"""
    if not item.product.stackable:
        # Check if anything would be placed on top
        x, y, z = item.position
        l, w, h = item.product.dimensions
        
        for existing_item in container.packed_items:
            ex, ey, ez = existing_item.position
            if (ex < x + l and ex + existing_item.product.dimensions[0] > x and
                ey < y + w and ey + existing_item.product.dimensions[1] > y and
                ez > z and ez < z + h):
                return False
    return True

def find_best_position(container: SmartContainer, product: ProductSpecs) -> Optional[Tuple[float, float, float]]:
    """Improved position finding with better stability handling"""
    best_pos = None
    min_z = float('inf')
    best_rotation = 0
    
    # Start from bottom-up with smaller increments for better precision
    increment = 0.05  # Reduced from 0.1 for finer positioning
    
    # Try all possible rotations for regular shapes
    rotations = [0, 90, 180, 270] if product.shape_type == ShapeType.REGULAR else [0]
    
    for rotation in rotations:
        l, w, h = product.dimensions
        if rotation in [90, 270]:
            l, w = w, l
            
        # Try to place near existing items or walls for better support
        existing_positions = [(0, 0)]  # Start with corner
        
        # Add positions next to existing items
        for item in container.packed_items:
            ix, iy, iz = item.position
            il, iw, _ = item.product.dimensions
            
            # Add positions next to item edges
            existing_positions.extend([
                (ix + il, iy),
                (ix, iy + iw),
                (ix, iy)
            ])
        
        # Filter valid positions
        existing_positions = [
            (x, y) for x, y in existing_positions 
            if x + l <= container.dimensions[0] and y + w <= container.dimensions[1]
        ]
        
        # Try positions near existing items first
        for x, y in existing_positions:
            # Find stable height at this position
            z = find_stable_height(container, product, (x, y, 0), rotation)
            
            if z is not None and z + h <= container.dimensions[2]:
                position = (x, y, z)
                
                # Create temporary item to check stability
                temp_item = PackedItem(product, position, rotation)
                
                if container.is_stable(temp_item) and z < min_z:
                    min_z = z
                    best_pos = position
                    best_rotation = rotation
        
        # If no stable position found near items, try grid positions
        if best_pos is None:
            for x in np.arange(0, container.dimensions[0] - l + increment, increment):
                for y in np.arange(0, container.dimensions[1] - w + increment, increment):
                    z = find_stable_height(container, product, (x, y, 0), rotation)
                    
                    if z is not None and z + h <= container.dimensions[2]:
                        position = (x, y, z)
                        temp_item = PackedItem(product, position, rotation)
                        
                        if container.is_stable(temp_item) and z < min_z:
                            min_z = z
                            best_pos = position
                            best_rotation = rotation
    
    if best_pos:
        product.rotation = best_rotation  # Store the best rotation
    return best_pos

def create_box_mesh(x: float, y: float, z: float, l: float, w: float, h: float):
    """Create proper box vertices and faces"""
    vertices = [
        # Bottom face
        [x, y, z], [x+l, y, z], [x+l, y+w, z], [x, y+w, z],
        # Top face
        [x, y, z+h], [x+l, y, z+h], [x+l, y+w, z+h], [x, y+w, z+h]
    ]
    
    # Define faces using vertex indices
    faces = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 2, 6], [1, 6, 5]   # right
    ]
    
    return np.array(vertices), np.array(faces)

def visualize_container(container: SmartContainer, failed_items: List[Tuple[ProductSpecs, str]]):
    """Enhanced visualization with better organization of failed items"""
    fig = go.Figure()

    # Add container outline
    fig.add_trace(go.Mesh3d(
        x=[0, container.dimensions[0], container.dimensions[0], 0, 0, 
           container.dimensions[0], container.dimensions[0], 0],
        y=[0, 0, container.dimensions[1], container.dimensions[1], 0, 0, 
           container.dimensions[1], container.dimensions[1]],
        z=[0, 0, 0, 0, container.dimensions[2], container.dimensions[2], 
           container.dimensions[2], container.dimensions[2]],
        i=[0, 0, 0, 0],
        j=[1, 2, 3, 4],
        k=[2, 3, 4, 1],
        opacity=0.1,
        color='gray',
        name="Container"
    ))

    # Add packed items with proper box visualization
    for item in container.packed_items:
        x, y, z = item.position
        l, w, h = item.product.dimensions
        if item.rotation in [90, 270]:
            l, w = w, l

        vertices, faces = create_box_mesh(x, y, z, l, w, h)
        
        # Create the box faces
        i = faces[:, 0]
        j = faces[:, 1]
        k = faces[:, 2]

        # Add box mesh
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i, j=j, k=k,
            color=item.color,
            opacity=0.7,
            name=f"{item.product.name} (Priority: {item.product.priority})",
            hoverinfo='text',
            hovertext=f"{item.product.name}<br>" +
                     f"Weight: {item.product.weight}kg<br>" +
                     f"Dims: {l:.2f}x{w:.2f}x{h:.2f}m"
        ))

        # Add edges for better visibility
        edges = create_box_edges(x, y, z, l, w, h)
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ))

    # Add failed items to the side
    side_spacing = 0.5  # Space between failed items
    current_x = container.dimensions[0] + 1  # Start 1m to the right of container
    current_y = 0
    current_z = 0
    max_height = 0

    for product, error in failed_items:
        l, w, h = product.dimensions
        
        # Check if we need to start a new row
        if current_y + w > container.dimensions[1]:
            current_y = 0
            current_x += max_height + side_spacing
            max_height = 0

        vertices, faces = create_box_mesh(current_x, current_y, current_z, l, w, h)
        
        # Add box mesh for failed items
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='rgba(255, 0, 0, 0.3)',  # Red transparent color for failed items
            opacity=0.5,
            name=f"✗ {product.name}",
            hoverinfo='text',
            hovertext=f"FAILED: {product.name}<br>" +
                     f"Error: {error}<br>" +
                     f"Weight: {product.weight}kg<br>" +
                     f"Dims: {l:.2f}x{w:.2f}x{h:.2f}m"
        ))

        current_y += w + side_spacing
        max_height = max(max_height, h)

    # Update layout with extended viewing area
    max_x = max(current_x + 2, container.dimensions[0] + 2)
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                eye=dict(x=max_x/2, y=max_x/2, z=max_x/2)
            ),
            xaxis=dict(range=[-1, max_x]),
            yaxis=dict(range=[-1, container.dimensions[1] + 1]),
            zaxis=dict(range=[-1, container.dimensions[2] + 1])
        ),
        title=f"Container Loading Visualization<br>" +
              f"Space Utilization: {container.calculate_space_utilization():.1f}%<br>" +
              f"Weight Utilization: {container.calculate_weight_utilization():.1f}%<br>" +
              f"Packed: {len(container.packed_items)} | Failed: {len(failed_items)}"
    )

    return fig

def create_box_edges(x: float, y: float, z: float, l: float, w: float, h: float):
    """Create edges for box visualization"""
    edges = []
    # Bottom edges
    edges.append(([x, x+l], [y, y], [z, z]))
    edges.append(([x+l, x+l], [y, y+w], [z, z]))
    edges.append(([x+l, x], [y+w, y+w], [z, z]))
    edges.append(([x, x], [y+w, y], [z, z]))
    # Top edges
    edges.append(([x, x+l], [y, y], [z+h, z+h]))
    edges.append(([x+l, x+l], [y, y+w], [z+h, z+h]))
    edges.append(([x+l, x], [y+w, y+w], [z+h, z+h]))
    edges.append(([x, x], [y+w, y], [z+h, z+h]))
    # Vertical edges
    edges.append(([x, x], [y, y], [z, z+h]))
    edges.append(([x+l, x+l], [y, y], [z, z+h]))
    edges.append(([x+l, x+l], [y+w, y+w], [z, z+h]))
    edges.append(([x, x], [y+w, y+w], [z, z+h]))
    return edges

def update_layer_information(container: SmartContainer, packed_item: PackedItem):
    """Update layer information after placing an item"""
    item_height = packed_item.position[2] + packed_item.product.dimensions[2]
    
    # Find or create appropriate layer
    for layer in container.layers:
        if abs(layer.height - item_height) < 0.01:
            layer.items.append(packed_item)
            layer.weight += packed_item.product.weight
            return
    
    # Create new layer if needed
    new_layer = Layer(item_height, [packed_item])
    container.layers.append(new_layer)
    container.layers.sort(key=lambda x: x.height)

def calculate_space_utilization(self) -> float:
    """Calculate container space utilization"""
    total_volume = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    used_volume = sum(item.product.dimensions[0] * 
                     item.product.dimensions[1] * 
                     item.product.dimensions[2] 
                     for item in self.packed_items)
    return (used_volume / total_volume) * 100

def calculate_weight_utilization(self) -> float:
    """Calculate container weight utilization"""
    total_weight = sum(item.product.weight for item in self.packed_items)
    return (total_weight / self.max_weight) * 100

def get_packing_statistics(self) -> Dict[str, float]:
    """Get detailed packing statistics"""
    return {
        "space_utilization": self.calculate_space_utilization(),
        "weight_utilization": self.calculate_weight_utilization(),
        "items_packed": len(self.packed_items),
        "average_height": np.mean([item.position[2] + item.product.dimensions[2] 
                                 for item in self.packed_items]) if self.packed_items else 0,
        "weight_balance_score": self.calculate_weight_balance_score()
    }

def calculate_weight_balance_score(self) -> float:
    """Calculate how well the weight is balanced (0-100)"""
    if not self.packed_items:
        return 100.0
    
    weight_grid = np.zeros((10, 10))
    for item in self.packed_items:
        x, y, _ = item.position
        grid_x = int((x / self.dimensions[0]) * 9)
        grid_y = int((y / self.dimensions[1]) * 9)
        weight_grid[grid_x, grid_y] += item.product.weight
    
    max_weight = np.max(weight_grid)
    min_weight = np.min(weight_grid[weight_grid > 0]) if np.any(weight_grid > 0) else 0
    return 100 * (1 - (max_weight - min_weight) / self.max_weight)

# Add these methods to the SmartContainer class
SmartContainer.calculate_space_utilization = calculate_space_utilization
SmartContainer.calculate_weight_utilization = calculate_weight_utilization
SmartContainer.get_packing_statistics = get_packing_statistics
SmartContainer.calculate_weight_balance_score = calculate_weight_balance_score

def group_similar_products(products: List[ProductSpecs]) -> List[List[ProductSpecs]]:
    """Group similar products together for better packing using indices"""
    groups = []
    used_indices = set()
    
    for i, p1 in enumerate(products):
        if i not in used_indices:
            current_group = [p1]
            used_indices.add(i)
            
            for j, p2 in enumerate(products):
                if j not in used_indices:
                    # Check if products are similar
                    if (abs(p1.dimensions[0] - p2.dimensions[0]) < 0.1 and
                        abs(p1.dimensions[1] - p2.dimensions[1]) < 0.1 and
                        abs(p1.dimensions[2] - p2.dimensions[2]) < 0.1 and
                        p1.fragility == p2.fragility and
                        p1.stackable == p2.stackable):
                        current_group.append(p2)
                        used_indices.add(j)
            
            groups.append(current_group)
    
    return groups

def split_oversized_products(products: List[ProductSpecs]) -> List[ProductSpecs]:
    """Split large products into smaller units if possible"""
    split_products = []
    
    for product in products:
        # Calculate volume
        volume = product.dimensions[0] * product.dimensions[1] * product.dimensions[2]
        
        if volume > 4 and product.quantity > 1:  # If product is large and multiple quantities
            # Try to split into smaller units
            new_dims = (
                product.dimensions[0] / 2 if product.dimensions[0] > 2 else product.dimensions[0],
                product.dimensions[1] / 2 if product.dimensions[1] > 2 else product.dimensions[1],
                product.dimensions[2]
            )
            
            # Create two products with half quantities
            half_qty = product.quantity // 2
            remainder = product.quantity % 2
            
            if half_qty > 0:
                split_products.append(ProductSpecs(
                    name=f"{product.name} (Part 1)",
                    dimensions=new_dims,
                    weight=product.weight / 2,
                    quantity=half_qty,
                    fragility=product.fragility,
                    stackable=product.stackable,
                    shape_type=product.shape_type,
                    boxing_type=product.boxing_type,
                    environment=product.environment.copy(),
                    priority=product.priority
                ))
                
                split_products.append(ProductSpecs(
                    name=f"{product.name} (Part 2)",
                    dimensions=new_dims,
                    weight=product.weight / 2,
                    quantity=half_qty + remainder,
                    fragility=product.fragility,
                    stackable=product.stackable,
                    shape_type=product.shape_type,
                    boxing_type=product.boxing_type,
                    environment=product.environment.copy(),
                    priority=product.priority
                ))
        else:
            split_products.append(product)
    
    return split_products

def create_3d_visualization(container: SmartContainer, failed_items: List[Tuple[ProductSpecs, str]]):
    """Create a more realistic container visualization"""
    fig = go.Figure()

    # Create container walls with proper thickness
    wall_thickness = 0.05
    walls = [
        # Floor
        {'pos': [0, 0, 0], 'size': [container.dimensions[0], container.dimensions[1], wall_thickness], 'color': 'lightgray'},
        # Left wall
        {'pos': [0, 0, 0], 'size': [wall_thickness, container.dimensions[1], container.dimensions[2]], 'color': 'lightgray'},
        # Right wall
        {'pos': [container.dimensions[0]-wall_thickness, 0, 0], 'size': [wall_thickness, container.dimensions[1], container.dimensions[2]], 'color': 'lightgray'},
        # Back wall
        {'pos': [0, container.dimensions[1]-wall_thickness, 0], 'size': [container.dimensions[0], wall_thickness, container.dimensions[2]], 'color': 'lightgray'},
    ]

    # Add container walls
    for wall in walls:
        x, y, z = wall['pos']
        l, w, h = wall['size']
        vertices, faces = create_box_mesh(x, y, z, l, w, h)
        
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=wall['color'],
            opacity=0.4,
            hoverinfo='none',
            showlegend=False
        ))

    # Create loading zones grid
    grid_size = 0.2
    for x in np.arange(0, container.dimensions[0], grid_size):
        for y in np.arange(0, container.dimensions[1], grid_size):
            fig.add_trace(go.Scatter3d(
                x=[x, x], y=[y, y],
                z=[0, 0.01],
                mode='lines',
                line=dict(color='lightgray', width=1),
                showlegend=False,
                hoverinfo='none'
            ))

    # Add packed items with proper spacing and orientation
    color_scale = px.colors.qualitative.Set3
    for idx, item in enumerate(container.packed_items):
        x, y, z = item.position
        l, w, h = item.product.dimensions
        if item.rotation in [90, 270]:
            l, w = w, l

        color = color_scale[idx % len(color_scale)]
        
        # Add item with proper edges and faces
        vertices, faces = create_box_mesh(x, y, z, l, w, h)
        
        # Add main box
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color,
            opacity=0.7,
            name=f"{item.product.name}",
            hovertext=f"Item: {item.product.name}<br>" +
                     f"Weight: {item.product.weight}kg<br>" +
                     f"Dims: {l:.2f}x{w:.2f}x{h:.2f}m<br>" +
                     f"Priority: {item.product.priority}<br>" +
                     f"Fragility: {item.product.fragility.name}",
            hoverinfo='text'
        ))

        # Add box edges for better visibility
        add_box_edges(fig, x, y, z, l, w, h, color='black')

    # Add failed items section
    if failed_items:
        section_start = container.dimensions[0] + 1
        current_x = section_start
        current_y = 0
        current_z = 0
        max_height = 0

        # Add section title
        fig.add_annotation(
            x=section_start,
            y=container.dimensions[1]/2,
            text="Failed Items",
            showarrow=False,
            font=dict(size=14, color='red')
        )

        # Add failed items
        for product, error in failed_items:
            l, w, h = product.dimensions
            
            if current_y + w > container.dimensions[1]:
                current_y = 0
                current_x += max_height + 0.5
                max_height = 0

            vertices, faces = create_box_mesh(current_x, current_y, current_z, l, w, h)
            
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='red',
                opacity=0.3,
                name=f"Failed: {product.name}",
                hovertext=f"Failed Item: {product.name}<br>" +
                         f"Error: {error}<br>" +
                         f"Dims: {l:.2f}x{w:.2f}x{h:.2f}m",
                hoverinfo='text'
            ))

            current_y += w + 0.3
            max_height = max(max_height, h)

    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis_title="Length (m)",
            yaxis_title="Width (m)",
            zaxis_title="Height (m)",
        ),
        title=dict(
            text=f"Container Loading Plan<br>" +
                 f"Utilization: {container.calculate_space_utilization():.1f}% | " +
                 f"Items Packed: {len(container.packed_items)}/{len(container.packed_items) + len(failed_items)}",
            y=0.95
        ),
        margin=dict(l=0, r=0, t=100, b=0)
    )

    return fig

def add_box_edges(fig, x, y, z, l, w, h, color='black'):
    """Add edges to a box for better visibility"""
    lines = [
        # Bottom edges
        [[x, x+l], [y, y], [z, z]],
        [[x+l, x+l], [y, y+w], [z, z]],
        [[x, x], [y, y+w], [z, z]],
        [[x, x+l], [y+w, y+w], [z, z]],
        # Top edges
        [[x, x+l], [y, y], [z+h, z+h]],
        [[x+l, x+l], [y, y+w], [z+h, z+h]],
        [[x, x], [y, y+w], [z+h, z+h]],
        [[x, x+l], [y+w, y+w], [z+h, z+h]],
        # Vertical edges
        [[x, x], [y, y], [z, z+h]],
        [[x+l, x+l], [y, y], [z, z+h]],
        [[x+l, x+l], [y+w, y+w], [z, z+h]],
        [[x, x], [y+w, y+w], [z, z+h]]
    ]

    for line in lines:
        fig.add_trace(go.Scatter3d(
            x=line[0], y=line[1], z=line[2],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo='none'
        ))

# Example usage
if __name__ == "__main__":
    try:
        print("Loading products from CSV...")
        original_products = load_products_from_csv('products.csv')
        
        print("\nCreating container...")
        container = SmartContainer(6.06, 2.44, 2.59, 28000)
        
        # First try with original products
        print("\nAttempting to pack items...")
        success, messages, failed_items = pack_items(container, original_products)
        
        # If first attempt fails, try with split products
        if not success:
            print("\nFirst attempt failed. Trying with split products...")
            split_products = split_oversized_products(original_products)
            container = SmartContainer(6.06, 2.44, 2.59, 28000)  # Reset container
            success, messages, failed_items = pack_items(container, split_products)
        
        print("\nCreating visualization...")
        fig = create_3d_visualization(container, failed_items)
        fig.show()
        
        # Print final statistics
        print("\nFinal Results:")
        print(f"Total items attempted: {sum(p.quantity for p in original_products)}")
        print(f"Successfully packed: {len(container.packed_items)}")
        print(f"Failed to pack: {len(failed_items)}")
        
        if failed_items:
            print("\nFailed Items:")
            for product, error in failed_items:
                print(f"- {product.name}: {error}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()

