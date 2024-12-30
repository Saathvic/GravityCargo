import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import math
import plotly.graph_objects as go
from typing import List, Tuple, Dict
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
from dash import dash_table
from datetime import datetime
import json

# Expanded container dimensions (Length, Width, Height) in meters
CONTAINER_TYPES = {
    # Standard Shipping Containers
    'Twenty-foot': (6.06, 2.44, 2.59),
    'Forty-foot': (12.01, 2.44, 2.59),
    'Forty-foot-high-cube': (12.01, 2.44, 2.89),
    
    # Air Cargo Containers
    'Air-LD3': (1.53, 1.53, 1.63),
    'Air-LD7': (3.18, 2.44, 1.63),
    'Air-LD11': (3.18, 2.44, 2.44),
    'Air-M1': (2.44, 1.63, 1.63),
    
    # Road Transport
    'Truck-Standard': (13.6, 2.48, 2.7),
    'Truck-Medium': (8.5, 2.48, 2.7),
    'Truck-Small': (6.1, 2.44, 2.6),
    'Van-Large': (4.3, 1.78, 1.9),
    'Van-Medium': (3.1, 1.78, 1.9),
    
    # Rail Transport
    'Rail-Box-Car': (18.29, 3.05, 3.05),
    'Rail-Container-Car': (12.19, 2.44, 2.59),
    'Rail-Refrigerated': (18.29, 2.95, 2.95),
    
    # Special Containers
    'Reefer-20ft': (5.46, 2.29, 2.27),
    'Reefer-40ft': (11.58, 2.29, 2.27),
    'Open-Top-20ft': (6.06, 2.44, 2.59),
    'Flat-Rack-40ft': (12.01, 2.44, 2.28)
}

# Updated transport modes with new container options
TRANSPORT_MODES = {
    '1': ('Sea Transport', [
        'Twenty-foot', 'Forty-foot', 'Forty-foot-high-cube',
        'Reefer-20ft', 'Reefer-40ft', 'Open-Top-20ft', 'Flat-Rack-40ft'
    ]),
    '2': ('Air Transport', [
        'Air-LD3', 'Air-LD7', 'Air-LD11', 'Air-M1'
    ]),
    '3': ('Road Transport', [
        'Truck-Standard', 'Truck-Medium', 'Truck-Small',
        'Van-Large', 'Van-Medium'
    ]),
    '4': ('Rail Transport', [
        'Rail-Box-Car', 'Rail-Container-Car', 'Rail-Refrigerated'
    ]),
    '5': ('Custom Dimensions', [])
}

def get_transport_config():
    print("\n=== Transport Mode Selection ===")
    for key, (mode, _) in TRANSPORT_MODES.items():
        print(f"{key}. {mode}")
    
    while True:
        mode_choice = input("\nSelect transport mode (1-5): ")
        if mode_choice in TRANSPORT_MODES:
            break
        print("Invalid choice. Please try again.")
    
    mode_name, container_options = TRANSPORT_MODES[mode_choice]
    
    if mode_choice == '5':
        # Custom dimensions
        print("\nEnter custom container dimensions (in meters):")
        length = float(input("Length: "))
        width = float(input("Width: "))
        height = float(input("Height: "))
        return (length, width, height)
    else:
        print(f"\nAvailable {mode_name} container types:")
        print("\nID | Type | Dimensions (L × W × H)")
        print("-" * 50)
        for i, container in enumerate(container_options, 1):
            dims = CONTAINER_TYPES[container]
            print(f"{i:2d} | {container:20s} | {dims[0]:.2f}m × {dims[1]:.2f}m × {dims[2]:.2f}m")
        
        while True:
            try:
                choice = int(input(f"\nSelect container type (1-{len(container_options)}): "))
                if 1 <= choice <= len(container_options):
                    return CONTAINER_TYPES[container_options[choice-1]]
            except ValueError:
                pass
            print("Invalid choice. Please try again.")

class Item:
    def __init__(self, name, length, width, height, weight, quantity, fragility, stackable, boxing_type, bundle):
        self.name = name
        self.original_dims = (float(length), float(width), float(height))
        self.weight = float(weight)
        self.quantity = int(quantity)  # Ensure quantity is integer
        self.fragility = fragility
        self.stackable = stackable
        self.boxing_type = boxing_type
        self.bundle = bundle
        self.position = None
        self.items_above = []
        self.color = f'rgb({random.randint(50,200)},{random.randint(50,200)},{random.randint(50,200)})'
        
        # Calculate dimensions with smarter bundling
        if bundle == 'YES' and self.quantity > 1:  # Use self.quantity after conversion
            self.dimensions = self._calculate_bundle_dimensions()
            self.weight = self.weight * self.quantity  # Use converted values
        else:
            self.dimensions = self.original_dims

    def _calculate_bundle_dimensions(self) -> Tuple[float, float, float]:
        """Calculate optimal bundle dimensions considering container constraints"""
        orig_l, orig_w, orig_h = self.original_dims
        qty = int(self.quantity)  # Ensure integer quantity
        
        # Maximum container dimensions to respect
        max_length = 13.0  # Slightly less than typical container length
        max_width = 2.4    # Standard container width
        max_height = 2.4   # Standard container height
        
        # Find best arrangement that respects container dimensions
        best_arrangement = None
        best_score = float('inf')  # Lower score is better
        
        # Try different arrangements
        for x in range(1, qty + 1):
            for y in range(1, qty + 1):
                z = -(-qty // (x * y))  # Ceiling division
                
                # Calculate dimensions for this arrangement
                length = orig_l * x
                width = orig_w * y
                height = orig_h * z
                
                # Skip if any dimension exceeds container limits
                if width > max_width or height > max_height or length > max_length:
                    continue
                    
                # Calculate score (prefer lower height and width over length)
                score = (height * 3) + (width * 2) + length
                
                # Check if this arrangement is complete and better than current best
                if x * y * z >= qty and score < best_score:
                    best_score = score
                    best_arrangement = (x, y, z)
        
        if best_arrangement:
            x, y, z = best_arrangement
            return (orig_l * x, orig_w * y, orig_h * z)
            
        # If no valid arrangement found, try to minimize height and width
        area_needed = orig_l * orig_w * qty
        max_layers = int(max_height / orig_h)
        min_layers = max(1, -(-qty // int((max_length * max_width) / (orig_l * orig_w))))
        
        for layers in range(min_layers, max_layers + 1):
            items_per_layer = -(-qty // layers)
            # Try to arrange items in each layer
            width_count = int(max_width / orig_w)
            length_count = -(-items_per_layer // width_count)
            
            if length_count * orig_l <= max_length:
                return (orig_l * length_count, orig_w * width_count, orig_h * layers)
        
        # If still no solution, return minimal stacking arrangement
        return (orig_l, orig_w, orig_h * min(qty, max_layers))

    def _check_overlap_2d(self, rect1: Tuple[float, float, float, float], 
                        rect2: Tuple[float, float, float, float]) -> bool:
        """Check if two rectangles overlap in 2D"""
        x1, y1, w1, d1 = rect1
        x2, y2, w2, d2 = rect2
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                   y1 + d1 <= y2 or y2 + d2 <= y1)

class MaximalSpace:
    def __init__(self, x, y, z, width, height, depth):
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth

    def get_volume(self):
        return self.width * self.height * self.depth

    def can_fit_item(self, item_dims):
        return (self.width >= item_dims[0] and 
                self.height >= item_dims[1] and 
                self.depth >= item_dims[2])

class EnhancedContainer:
    def __init__(self, dimensions):
        # Validate dimensions
        if not all(isinstance(d, (int, float)) and d > 0 for d in dimensions):
            raise ValueError("Container dimensions must be positive numbers")
        if len(dimensions) != 3:
            raise ValueError("Container must have exactly 3 dimensions (length, width, height)")
            
        self.dimensions = tuple(float(d) for d in dimensions)
        self.items = []
        self.spaces = [MaximalSpace(0, 0, 0, dimensions[0], dimensions[1], dimensions[2])]
        self.weight_distribution = {}
        self.volume_utilization = 0.0
        self.layer_height = 0
        self.weight_map = np.zeros((10, 10))  # Grid for weight distribution
        self.center_of_gravity = [0, 0, 0]
        self.unpacked_reasons = {}
        self.total_weight = 0
        self.unused_spaces = []  # Track remaining spaces
        self.unpacked_reasons = {}  # Enhanced reasons tracking
        self.total_volume = dimensions[0] * dimensions[1] * dimensions[2]
        self.remaining_volume = self.total_volume
        self.support_mechanisms = []  # Track support mechanisms used

    def _get_valid_rotations(self, item):
        """Get all valid rotations considering container constraints"""
        rotations = []
        l, w, h = item.dimensions
        
        # Base orientations
        possible_rotations = [
            (l, w, h), (l, h, w),
            (w, l, h), (w, h, l),
            (h, l, w), (h, w, l)
        ]
        
        # Filter rotations based on container dimensions and item properties
        for rot in possible_rotations:
            if (all(d <= max_d for d, max_d in zip(rot, self.dimensions)) and  # Check container fit
                (item.fragility != 'HIGH' or rot[2] <= item.dimensions[2])):   # Check fragility constraints
                rotations.append(rot)
        
        return rotations

    def _find_best_position(self, item, rotations):
        best_score = float('-inf')
        best_pos = None
        best_rot = None
        best_space = None

        for rotation in rotations:
            for space in self.spaces:
                if space.can_fit_item(rotation):
                    # Try different positions within the space
                    positions = [
                        (space.x, space.y, space.z),  # Bottom-front-left
                        (space.x, space.y + space.depth - rotation[1], space.z),  # Bottom-back-left
                        (space.x + space.width - rotation[0], space.y, space.z),  # Bottom-front-right
                    ]

                    for pos in positions:
                        score = self._evaluate_position(item, pos, rotation)
                        if score > best_score and self._is_valid_placement(item, pos, rotation):
                            best_score = score
                            best_pos = pos
                            best_rot = rotation
                            best_space = space

        return best_pos, best_rot, best_space

    def _evaluate_position(self, item, pos, dims):
        x, y, z = pos
        score = 0
        
        # Strongly prefer lower positions to encourage stacking
        score -= z * 5
        
        # Prefer positions that maximize vertical space usage
        vertical_efficiency = z / self.dimensions[2]
        score += (1 - vertical_efficiency) * 4
        
        # Prefer positions closer to walls and other items
        wall_bonus = self._calculate_wall_contact(pos, dims)
        score += wall_bonus * 3
        
        # Original scoring components with adjusted weights
        score += self._calculate_stability_score(item, pos, dims) * 4.5
        score += self._calculate_local_density(pos, dims) * 3.5
        score += self._calculate_contact_score(item, pos, dims) * 3
        
        return score

    def _calculate_wall_contact(self, pos, dims):
        """Calculate how many container walls the item touches"""
        x, y, z = pos
        l, w, h = dims
        score = 0
        tolerance = 0.001
        
        # Check wall contacts
        if abs(x) < tolerance or abs(x + l - self.dimensions[0]) < tolerance:
            score += 1
        if abs(y) < tolerance or abs(y + w - self.dimensions[1]) < tolerance:
            score += 1
        if abs(z) < tolerance:  # Floor contact
            score += 2
            
        return score

    def _has_support(self, pos, dims):
        if pos[2] == 0:  # On the ground
            return True
            
        x, y, z = pos
        w, d, h = dims
        
        # Check if there's an item directly below
        for item in self.items:
            if (item.position[2] + item.dimensions[2] == z and
                self._check_overlap_2d(
                    (x, y, w, d),
                    (item.position[0], item.position[1], 
                     item.dimensions[0], item.dimensions[1])
                )):
                return True
        return False

    def _calculate_local_density(self, pos, dims):
        x, y, z = pos
        w, d, h = dims
        nearby_volume = 0
        total_volume = w * d * h * 27  # 3x3x3 grid around position
        
        for item in self.items:
            if (abs(item.position[0] - x) <= w * 2 and
                abs(item.position[1] - y) <= d * 2 and
                abs(item.position[2] - z) <= h * 2):
                nearby_volume += (item.dimensions[0] * 
                                item.dimensions[1] * 
                                item.dimensions[2])
        
        return nearby_volume / total_volume

    def _update_spaces(self, pos, dims, used_space):
        x, y, z = pos
        w, d, h = dims
        
        # Remove used space
        self.spaces.remove(used_space)
        
        # Generate new spaces
        new_spaces = []
        
        # Space above the item
        if used_space.height > h:
            new_spaces.append(MaximalSpace(
                x, y, z + h,
                w, used_space.height - h, d
            ))
            
        # Space to the right
        if used_space.width > w:
            new_spaces.append(MaximalSpace(
                x + w, y, z,
                used_space.width - w, used_space.height, used_space.depth
            ))
            
        # Space to the front
        if used_space.depth > d:
            new_spaces.append(MaximalSpace(
                x, y + d, z,
                used_space.width, used_space.height, used_space.depth - d
            ))
        
        # Add new spaces and merge overlapping ones
        self.spaces.extend(new_spaces)
        self._merge_spaces()

    def _merge_spaces(self):
        i = 0
        while i < len(self.spaces):
            j = i + 1
            while j < len(self.spaces):
                if self._can_merge_spaces(self.spaces[i], self.spaces[j]):
                    self.spaces[i] = self._merge_two_spaces(self.spaces[i], self.spaces[j])
                    self.spaces.pop(j)
                else:
                    j += 1
            i += 1

    def pack_items(self, items: List[Item]):
        """Fix the item quantity handling in packing"""
        expanded_items = []
        
        for item in items:
            if item.bundle == 'YES' and item.quantity > 1:
                # Handle bundled items - already processed in Item initialization
                expanded_items.append(item)
            else:
                # For non-bundled items, create individual copies
                try:
                    quantity = int(item.quantity)  # Ensure integer conversion
                    for i in range(quantity):
                        new_item = Item(
                            name=f"{item.name}_{i+1}",
                            length=float(item.original_dims[0]),
                            width=float(item.original_dims[1]),
                            height=float(item.original_dims[2]),
                            weight=float(item.weight),
                            quantity=1,
                            fragility=item.fragility,
                            stackable=item.stackable,
                            boxing_type=item.boxing_type,
                            bundle='NO'
                        )
                        expanded_items.append(new_item)
                except ValueError as e:
                    print(f"Error converting quantity for item {item.name}: {e}")
                    continue

        # Sort expanded items for optimal packing
        sorted_items = sorted(expanded_items, 
                            key=lambda x: (
                                -(x.dimensions[0] * x.dimensions[1]),  # Larger base area first
                                x.dimensions[2],                       # Lower height preferred
                                -x.weight                             # Heavier items first
                            ))

        # Pack each item
        for item in sorted_items:
            if not self._try_pack_in_layer(item, 0):
                # Try on existing layers
                packed = False
                for height in sorted(set(i.position[2] + i.dimensions[2] 
                                   for i in self.items if i.position)):
                    if self._try_pack_in_layer(item, height):
                        packed = True
                        break
                        
                if not packed:
                    base_name = item.name.rsplit('_', 1)[0] if '_' in item.name else item.name
                    self.unpacked_reasons[item.name] = (
                        self._get_unpacking_reason(item), item
                    )

        # Calculate volume utilization and total weight
        self._update_metrics()

    def _update_metrics(self):
        """Enhanced metrics calculation with error handling"""
        try:
            # Calculate packed volume safely
            packed_volume = sum(
                max(0, item.dimensions[0]) * max(0, item.dimensions[1]) * max(0, item.dimensions[2])
                for item in self.items
            )
            
            # Update metrics with bounds checking
            self.volume_utilization = min(100, (packed_volume / max(0.001, self.total_volume)) * 100)
            self.total_weight = sum(max(0, item.weight) for item in self.items)
            self.remaining_volume = max(0, self.total_volume - packed_volume)

            # Update center of gravity
            if self.items:
                self._update_center_of_gravity()
                
        except Exception as e:
            print(f"Error updating metrics: {str(e)}")
            # Set safe default values
            self.volume_utilization = 0
            self.total_weight = 0
            self.remaining_volume = self.total_volume

    def _try_pack_in_layer(self, item: Item, height: float) -> bool:
        """Enhanced packing with better error handling"""
        try:
            # Validate input parameters
            if not isinstance(item, Item):
                raise ValueError("Invalid item type")
            if not isinstance(height, (int, float)):
                raise ValueError("Invalid height value")
            if height < 0 or height > self.dimensions[2]:
                return False  # Height out of bounds

            # Get valid rotations with dimension checking
            rotations = [rot for rot in self._get_valid_rotations(item)
                        if all(d <= max_d for d, max_d in zip(rot, self.dimensions))]
            
            if not rotations:
                return False  # No valid rotation found

            best_pos = None
            best_rot = None
            best_score = float('-inf')
            best_space = None
            
            # For each rotation, try to find best position
            for rotation in rotations:
                # Skip if height + item height exceeds container height
                if height + rotation[2] > self.dimensions[2]:
                    continue

                # Find valid positions in current layer
                for space in self.spaces:
                    if not space.can_fit_item(rotation):
                        continue
                        
                    # Only consider spaces at the correct height
                    if abs(space.z - height) > 0.001:
                        continue
                        
                    pos = (space.x, space.y, height)
                    if self._is_valid_placement(item, pos, rotation):
                        score = self._evaluate_position_enhanced(item, pos, rotation)
                        if score > best_score:
                            best_score = score
                            best_pos = pos
                            best_rot = rotation
                            best_space = space
        
            if best_pos and best_rot and best_space:
                item.position = best_pos
                item.dimensions = best_rot
                self.items.append(item)
                self._update_spaces(best_pos, best_rot, best_space)
                self._update_weight_distribution(item)
                return True
            
            return False

        except Exception as e:
            print(f"Error packing item {item.name}: {str(e)}")
            return False

    def _evaluate_position_enhanced(self, item, pos, dims):
        """Enhanced position evaluation with reduced stability constraints"""
        score = 0
        x, y, z = pos
        w, d, h = dims
        
        # Reduced weight for support score
        support_score = self._calculate_support_score(item, pos, dims)
        score += support_score * 5  # Reduced from 10 to 5
        
        # Prefer positions against walls
        wall_contact = self._calculate_wall_contact(pos, dims)
        score += wall_contact * 5
        
        # Prefer positions that minimize gaps
        density_score = self._calculate_local_density(pos, dims)
        score += density_score * 4
        
        # Consider center of gravity
        cog_score = self._evaluate_cog_impact(item, pos)
        score += cog_score * 3
        
        # Consider load bearing capacity utilization
        if z > 0:
            items_below = self._get_items_below(pos, (w, d))
            stackable_support = all(item.stackable for item in items_below)
            score += 5 if stackable_support else -10
        
        return score

    def _find_valid_positions(self, item, rotation, height):
        """Find all valid positions at given height"""
        valid_positions = []
        step_size = 0.1  # Grid size for position search
        
        for x in np.arange(0, self.dimensions[0] - rotation[0] + step_size, step_size):
            for y in np.arange(0, self.dimensions[1] - rotation[1] + step_size, step_size):
                pos = (x, y, height)
                if self._is_valid_placement(item, pos, rotation):
                    valid_positions.append(pos)
        
        return valid_positions

    def _calculate_support_score(self, item, pos, dims):
        """Calculate support score with reduced constraints"""
        x, y, z = pos
        w, d, h = dims
        
        if z == 0:  # On the ground
            return 1.0
            
        support_area = 0
        total_area = w * d
        items_below = self._get_items_below(pos, (w, d))
        
        for below_item in items_below:
            overlap = self._calculate_overlap_area(
                (x, y, w, d),
                (below_item.position[0], below_item.position[1],
                 below_item.dimensions[0], below_item.dimensions[1])
            )
            # Relaxed load bearing requirement
            stackable_support = all(item.stackable for item in items_below)
            support_area += overlap * stackable_support
        
        # If support area is insufficient, add support mechanisms
        support_ratio = support_area / total_area
        if support_ratio < 0.3:  # Reduced from 0.5
            if self._can_add_support(pos, dims):
                self._add_support_mechanism(pos, dims)
                return 0.8  # Good score with support mechanism
        
        return max(0.3, support_ratio)  # Accept lower support ratios

    def _can_add_support(self, pos, dims):
        """Check if we can add support mechanisms"""
        # Check if there's space for supports
        x, y, z = pos
        w, d, h = dims
        
        # Simple space check for now
        return z < (self.dimensions[2] * 0.8)  # Allow support if not too high

    def _evaluate_stability(self, item, pos, dims):
        """Evaluate stability with relaxed constraints"""
        score = 0
        x, y, z = pos
        w, d, h = dims

        # Base support score (reduced weight)
        support_score = self._calculate_support_score(item, pos, dims)
        score += support_score * 3  # Reduced from 5

        # Increased weight for wall contact
        wall_contact = self._calculate_wall_contact(pos, dims)
        score += wall_contact * 4  # Increased from 3

        # Consider item's own properties
        if item.fragility == 'LOW':
            score += 2  # Bonus for sturdy items
        
        # Prefer lower heights for heavy items
        if item.weight > 500:  # Heavy item
            height_factor = 1 - (z / self.dimensions[2])
            score += height_factor * 2

        return score

    def _calculate_wall_contact(self, pos, dims):
        """Calculate how many container walls the item touches"""
        x, y, z = pos
        l, w, h = dims
        score = 0
        tolerance = 0.001
        
        # Check wall contacts
        if abs(x) < tolerance or abs(x + l - self.dimensions[0]) < tolerance:
            score += 1
        if abs(y) < tolerance or abs(y + w - self.dimensions[1]) < tolerance:
            score += 1
        if abs(z) < tolerance:  # Floor contact
            score += 2
            
        return score

    def _add_support_mechanism(self, pos, dims):
        """Add support mechanisms to stabilize items"""
        x, y, z = pos
        w, d, h = dims
        self.support_mechanisms.append({
            'position': (x, y, z),
            'dimensions': (w, d, h),
            'type': 'block'  # Example support mechanism type
        })

    def _calculate_weight_balance(self, x: float, weight: float) -> float:
        """Calculate weight balance score for the position"""
        section_size = self.dimensions[0] / 3
        current_section = int(x / section_size)
        
        # Update weight map
        self.weight_map[current_section] += weight
        
        # Calculate balance score
        total_weight = sum(self.weight_map.flatten())
        if total_weight == 0:
            return 1.0
            
        balance = 1.0 - (max(self.weight_map.flatten()) / total_weight - 1/3)
        return balance

    def _check_overlap_2d(self, rect1: Tuple[float, float, float, float], 
                         rect2: Tuple[float, float, float, float]) -> bool:
        """Check if two rectangles overlap in 2D"""
        x1, y1, w1, d1 = rect1
        x2, y2, w2, d2 = rect2  # Fixed: Added missing d2
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                   y1 + d1 <= y2 or y2 + d2 <= y1)

    def _can_merge_spaces(self, s1: MaximalSpace, s2: MaximalSpace) -> bool:
        """Check if two spaces can be merged"""
        # Check if spaces are adjacent and have same dimensions in two directions
        return ((s1.x + s1.width == s2.x or s2.x + s2.width == s1.x) and
                s1.y == s2.y and s1.z == s2.z and
                s1.height == s2.height and s1.depth == s2.depth)

    def _merge_two_spaces(self, s1: MaximalSpace, s2: MaximalSpace) -> MaximalSpace:
        """Merge two spaces into one"""
        x = min(s1.x, s2.x)
        width = s1.width + s2.width
        return MaximalSpace(x, s1.y, s1.z, width, s1.height, s1.depth)

    def _is_valid_placement(self, item: Item, pos: Tuple[float, float, float], 
                          dims: Tuple[float, float, float]) -> bool:
        """Check if placement is valid including stackability"""
        x, y, z = pos
        w, d, h = dims
        
        # Check container boundaries
        if (x + w > self.dimensions[0] or
            y + d > self.dimensions[1] or
            z + h > self.dimensions[2] or
            x < 0 or y < 0 or z < 0):
            return False
            
        # Check overlap with other items
        for placed_item in self.items:
            if self._check_overlap_3d(
                (x, y, z, w, d, h),
                (placed_item.position[0], placed_item.position[1], placed_item.position[2],
                 placed_item.dimensions[0], placed_item.dimensions[1], placed_item.dimensions[2])
            ):
                return False
                
        # Check stackability
        if z > 0:  # If not on the ground
            items_below = self._get_items_below((x, y, z), (w, d))
            if not items_below:
                return False  # Need support from below
            
            # Check if all items below are stackable
            for below_item in items_below:
                if not below_item.stackable:
                    return False  # Can't stack on non-stackable items
                
        return True

    def _get_items_below(self, pos: Tuple[float, float, float], 
                        dims: Tuple[float, float]) -> List[Item]:
        """Find items directly below the given position"""
        x, y, z = pos
        w, d = dims
        items_below = []
        
        for item in self.items:
            if (abs(item.position[2] + item.dimensions[2] - z) < 0.001 and
                self._check_overlap_2d(
                    (x, y, w, d),
                    (item.position[0], item.position[1], 
                     item.dimensions[0], item.dimensions[1])
                )):
                items_below.append(item)
                
        return items_below

    def _calculate_weight_above(self, item: Item) -> float:
        """Calculate total weight of items above the given item"""
        total_weight = 0
        item_top = item.position[2] + item.dimensions[2]
        
        for other in self.items:
            if (abs(other.position[2] - item_top) < 0.001 and
                self._check_overlap_2d(
                    (item.position[0], item.position[1], 
                     item.dimensions[0], item.dimensions[1]),
                    (other.position[0], other.position[1],
                     other.dimensions[0], other.dimensions[1])
                )):
                total_weight += other.weight
                total_weight += self._calculate_weight_above(other)
                
        return total_weight

    def _check_overlap_3d(self, box1: Tuple[float, float, float, float, float, float],
                         box2: Tuple[float, float, float, float, float, float]) -> bool:
        """Check if two boxes overlap in 3D"""
        x1, y1, z1, w1, d1, h1 = box1
        x2, y2, z2, w2, d2, h2 = box2  # Fixed: Added missing h2
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                   y1 + d1 <= y2 or y2 + d2 <= y1 or
                   z1 + h1 <= z2 or z2 + h2 <= z1)

    def _update_weight_distribution(self, item: Item) -> None:
        """Update weight distribution when placing a new item"""
        # Calculate which third of the container the item is in
        section_size = self.dimensions[0] / 3
        x = item.position[0]
        section = int(x / section_size)
        
        # Update weight distribution dictionary
        if section not in self.weight_distribution:
            self.weight_distribution[section] = 0
        self.weight_distribution[section] += item.weight
        
        # Update weight map for detailed balance calculation
        col = int((x / self.dimensions[0]) * self.weight_map.shape[1])
        row = int((item.position[1] / self.dimensions[1]) * self.weight_map.shape[0])
        if 0 <= row < self.weight_map.shape[0] and 0 <= col < self.weight_map.shape[1]:
            self.weight_map[row, col] += item.weight

    def _update_center_of_gravity(self):
        """Calculate center of gravity after each item placement"""
        total_moment = np.array([0.0, 0.0, 0.0])
        self.total_weight = 0
        
        for item in self.items:
            pos = np.array(item.position)
            center = pos + np.array(item.dimensions) / 2
            total_moment += center * item.weight
            self.total_weight += item.weight
            
        if self.total_weight > 0:
            self.center_of_gravity = total_moment / self.total_weight

    def _try_interlock_position(self, item, pos, rot):
        """Check if position creates good interlocking with nearby items"""
        x, y, z = pos
        contact_score = 0
        
        for placed_item in self.items:
            if self._has_surface_contact(pos, rot, placed_item):
                contact_score += 1
                
        return contact_score > 1  # Require contact with multiple items

    def _evaluate_cog_impact(self, item, pos):
        """Evaluate how an item placement affects center of gravity"""
        temp_cog = np.array(self.center_of_gravity)
        temp_weight = self.total_weight
        
        item_center = np.array(pos) + np.array(item.dimensions) / 2
        new_cog = (temp_cog * temp_weight + item_center * item.weight) / (temp_weight + item.weight)
        
        # Prefer positions that keep COG near center
        target = np.array(self.dimensions) / 2
        current_dist = np.linalg.norm(temp_cog - target)
        new_dist = np.linalg.norm(new_cog - target)
        
        return 1.0 / (1.0 + new_dist)

    def _has_surface_contact(self, pos1, dims1, item2) -> bool:
        """Check if two items have surface contact"""
        if not item2.position:  # Skip if item2 hasn't been placed
            return False
            
        x1, y1, z1 = pos1
        l1, w1, h1 = dims1
        x2, y2, z2 = item2.position
        l2, w2, h2 = item2.dimensions
        
        # Check for surface contact on each face with tolerance
        tolerance = 0.001
        
        # Bottom face contact
        if abs(z1 - (z2 + h2)) < tolerance:
            if self._check_overlap_2d(
                (x1, y1, l1, w1),
                (x2, y2, l2, w2)
            ):
                return True
                
        # Top face contact
        if abs((z1 + h1) - z2) < tolerance:
            if self._check_overlap_2d(
                (x1, y1, l1, w1),
                (x2, y2, l2, w2)
            ):
                return True
                
        # Front/back face contacts
        if abs(y1 - (y2 + w2)) < tolerance or abs((y1 + w1) - y2) < tolerance:
            if self._check_overlap_2d(
                (x1, z1, l1, h1),
                (x2, z2, l2, h2)
            ):
                return True
                
        # Left/right face contacts
        if abs(x1 - (x2 + l2)) < tolerance or abs((x1 + l1) - x2) < tolerance:
            if self._check_overlap_2d(
                (y1, z1, w1, h1),
                (y2, z2, w2, h2)
            ):
                return True
                
        return False

    def _calculate_weight_balance_score(self) -> float:
        """Calculate overall weight balance score"""
        if not self.weight_distribution:
            return 0.0
            
        weights = list(self.weight_distribution.values())
        total_weight = sum(weights)
        if total_weight == 0:
            return 1.0
            
        # Calculate variance in weight distribution
        mean_weight = total_weight / len(weights)
        variance = sum((w - mean_weight) ** 2 for w in weights) / len(weights)
        
        # Return normalized score (0-1, higher is better)
        return 1.0 / (1.0 + variance / 1000)

    def _calculate_interlocking_score(self) -> float:
        """Calculate overall interlocking score"""
        if not self.items:
            return 0.0
            
        total_contacts = 0
        for item in self.items:
            for other in self.items:
                if item != other and self._has_surface_contact(
                    item.position, item.dimensions, other
                ):
                    total_contacts += 1
                    
        # Return normalized score (0-1, higher is better)
        max_possible_contacts = len(self.items) * 6  # Each item can touch 6 sides
        return total_contacts / max_possible_contacts

    def _calculate_stability_score(self, item, pos, dims):
        """Calculate stability score for item placement"""
        x, y, z = pos
        w, d, h = dims
        score = 0
        
        # Check support from below
        support_area = 0
        total_area = w * d
        items_below = self._get_items_below(pos, (w, d))
        
        for below_item in items_below:
            overlap_area = self._calculate_overlap_area(
                (x, y, w, d),
                (below_item.position[0], below_item.position[1],
                 below_item.dimensions[0], below_item.dimensions[1])
            )
            support_area += overlap_area
        
        if z == 0:  # On the ground
            support_area = total_area
            
        support_ratio = support_area / total_area
        score += support_ratio * 5
        
        # Additional stability factors
        if support_ratio >= 0.8:  # Well supported
            score += 2
        if h <= max(w, d):  # Lower center of gravity
            score += 1
            
        return score

    def _calculate_contact_score(self, item, pos, dims):
        """Calculate score based on surface contact with other items"""
        score = 0
        for placed_item in self.items:
            if self._has_surface_contact(pos, dims, placed_item):
                # Higher score for larger contact areas
                contact_area = self._calculate_contact_area(pos, dims, placed_item)
                score += contact_area / (dims[0] * dims[1])
        return score

    def _calculate_overlap_area(self, rect1: Tuple[float, float, float, float], 
                          rect2: Tuple[float, float, float, float]) -> float:
        """Calculate overlap area between two rectangles"""
        x1, y1, w1, d1 = rect1
        x2, y2, w2, d2 = rect2
        
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + d1, y2 + d2) - max(y1, y2))
        
        return x_overlap * y_overlap

    def _calculate_contact_area(self, pos, dims, other_item) -> float:
        """Calculate the contact area between two items when they touch"""
        x1, y1, z1 = pos
        l1, w1, h1 = dims
        x2, y2, z2 = other_item.position
        l2, w2, h2 = other_item.dimensions
        
        total_contact_area = 0
        tolerance = 0.001
        
        # Bottom/Top face contact
        if abs(z1 - (z2 + h2)) < tolerance or abs((z1 + h1) - z2) < tolerance:
            overlap_area = self._calculate_overlap_area(
                (x1, y1, l1, w1),
                (x2, y2, l2, w2)
            )
            total_contact_area += overlap_area
            
        # Front/Back face contact
        if abs(y1 - (y2 + w2)) < tolerance or abs((y1 + w1) - y2) < tolerance:
            overlap_area = self._calculate_overlap_area(
                (x1, z1, l1, h1),
                (x2, z2, l2, h2)
            )
            total_contact_area += overlap_area
            
        # Left/Right face contact
        if abs(x1 - (x2 + l2)) < tolerance or abs((x1 + l1) - x2) < tolerance:
            overlap_area = self._calculate_overlap_area(
                (y1, z1, w1, h1),
                (y2, z2, w2, h2)
            )
            total_contact_area += overlap_area
            
        return total_contact_area

    def _update_unused_spaces(self):
        """Calculate and update remaining usable spaces"""
        used_volume = sum(item.dimensions[0] * item.dimensions[1] * item.dimensions[2] 
                         for item in self.items)
        self.remaining_volume = self.total_volume - used_volume
        
        # Find largest continuous spaces
        self.unused_spaces = self._find_empty_spaces()

    def _find_empty_spaces(self) -> List[Tuple[float, float, float, float, float, float]]:
        """Find largest empty spaces in container"""
        # Implementation of empty space detection algorithm
        empty_spaces = []
        min_space_size = 0.1  # Minimum useful space size in meters
        
        # Create grid of points to check for empty spaces
        x_points = sorted(set([0, self.dimensions[0]] + 
                            [item.position[0] for item in self.items] +
                            [item.position[0] + item.dimensions[0] for item in self.items]))
        y_points = sorted(set([0, self.dimensions[1]] + 
                            [item.position[1] for item in self.items] +
                            [item.position[1] + item.dimensions[1] for item in self.items]))
        z_points = sorted(set([0, self.dimensions[2]] + 
                            [item.position[2] for item in self.items] +
                            [item.position[2] + item.dimensions[2] for item in self.items]))
        
        # Check each potential space
        for i in range(len(x_points) - 1):
            for j in range(len(y_points) - 1):
                for k in range(len(z_points) - 1):
                    x = x_points[i]
                    y = y_points[j]
                    z = z_points[k]
                    w = x_points[i+1] - x
                    d = y_points[j+1] - y
                    h = z_points[k+1] - z
                    
                    # Check if space is empty and large enough
                    if (w >= min_space_size and d >= min_space_size and h >= min_space_size and
                        self._is_space_empty(x, y, z, w, d, h)):
                        empty_spaces.append((x, y, z, w, d, h))
        
        return empty_spaces

    def _is_space_empty(self, x, y, z, w, d, h):
        """Check if a given space is empty"""
        for item in self.items:
            if self._check_overlap_3d(
                (x, y, z, w, d, h),
                (item.position[0], item.position[1], item.position[2],
                 item.dimensions[0], item.dimensions[1], item.dimensions[2])
            ):
                return False
        return True

    def _get_unpacking_reason(self, item) -> str:
        """Get detailed reason why item couldn't be packed"""
        # Check dimensions
        if item.dimensions[0] > self.dimensions[0] or \
           item.dimensions[1] > self.dimensions[1] or \
           item.dimensions[2] > self.dimensions[2]:
            return (f"Dimensions too large: Item ({item.dimensions[0]:.2f}×{item.dimensions[1]:.2f}×{item.dimensions[2]:.2f}m) "
                    f"exceeds container ({self.dimensions[0]:.2f}×{self.dimensions[1]:.2f}×{self.dimensions[2]:.2f}m)")
        
        # Check volume availability
        item_volume = item.dimensions[0] * item.dimensions[1] * item.dimensions[2]
        if self.remaining_volume < item_volume:
            return f"Insufficient space: Need {item_volume:.2f}m³, only {self.remaining_volume:.2f}m³ available"
        
        # Check stability and support
        test_positions = []
        for space in self.spaces:
            if space.can_fit_item(item.dimensions):
                pos = (space.x, space.y, space.z)
                support_score = self._calculate_support_score(item, pos, item.dimensions)
                if support_score < 0.3:  # Our minimum threshold
                    test_positions.append({
                        'pos': pos,
                        'support_score': support_score,
                        'reason': f"Insufficient support (score: {support_score:.2f})"
                    })
                elif not self._check_stackability(item, pos):
                    test_positions.append({
                        'pos': pos,
                        'reason': f"Stackability exceeded (weight: {item.weight}kg)"
                    })
        
        if test_positions:
            reasons = [pos['reason'] for pos in test_positions]
            most_common_reason = max(set(reasons), key=reasons.count)
            return most_common_reason
        
        # Check available spaces
        max_space = max(self.unused_spaces, key=lambda s: s[3] * s[4] * s[5], default=None)
        if max_space:
            return (f"No valid position found. Best available space: "
                    f"{max_space[3]:.2f}×{max_space[4]:.2f}×{max_space[5]:.2f}m")
        
        return "No suitable position found due to combination of stability and space constraints"

    def generate_alternative_arrangement(self):
        """Generate an alternative packing arrangement"""
        # Reset container state
        self.items = []
        self.spaces = [MaximalSpace(0, 0, 0, self.dimensions[0], self.dimensions[1], self.dimensions[2])]
        self.weight_distribution = {}
        self.center_of_gravity = [0, 0, 0]
        
        # Modified sorting strategies for different arrangements
        strategies = [
            # Strategy 1: Volume
            lambda x: -(x.dimensions[0] * x.dimensions[1] * x.dimensions[2]),
            # Strategy 2: Height + Base Area
            lambda x: (x.dimensions[2], -(x.dimensions[0] * x.dimensions[1])),
            # Strategy 3: Weight + Load bearing
            lambda x: (-x.weight, -x.stackable)
        ]
        
        best_arrangement = None
        best_utilization = 0
        
        for strategy in strategies:
            # Reset container
            self.items = []
            self.spaces = [MaximalSpace(0, 0, 0, self.dimensions[0], self.dimensions[1], self.dimensions[2])]
            
            # Sort items using current strategy
            sorted_items = sorted(self.original_items, key=strategy)
            
            # Try to pack with current strategy
            for item in sorted_items:
                self._try_pack_item_with_enhanced_stacking(item)
            
            # Calculate utilization
            current_utilization = self._calculate_utilization()
            
            if current_utilization > best_utilization:
                best_utilization = current_utilization
                best_arrangement = self.items.copy()
        
        return best_arrangement

    def _try_pack_item_with_enhanced_stacking(self, item):
        """Try to pack item with enhanced stacking logic"""
        # Prefer positions that create stable stacks
        rotations = self._get_valid_rotations(item)
        best_pos = None
        best_score = float('-inf')
        
        for rotation in rotations:
            # Try to place on top of existing items first
            for placed_item in self.items:
                if placed_item.stackable:
                    pos = (placed_item.position[0], 
                          placed_item.position[1],
                          placed_item.position[2] + placed_item.dimensions[2])
                    
                    if self._is_valid_placement(item, pos, rotation):
                        score = self._evaluate_stacking_position(item, pos, rotation)
                        if score > best_score:
                            best_score = score
                            best_pos = (pos, rotation)
            
            # Try ground positions if no stack position found
            for space in self.spaces:
                if space.can_fit_item(rotation):
                    pos = (space.x, space.y, space.z)
                    if self._is_valid_placement(item, pos, rotation):
                        score = self._evaluate_stacking_position(item, pos, rotation)
                        if score > best_score:
                            best_score = score
                            best_pos = (pos, rotation)
        
        if best_pos:
            pos, rot = best_pos
            item.position = pos
            item.dimensions = rot
            self.items.append(item)
            self._update_spaces(pos, rot)
            return True
            
        return False

    def _evaluate_stacking_position(self, item, pos, dims):
        """Evaluate position with focus on efficient stacking"""
        score = 0
        x, y, z = pos
        
        # Strongly prefer lower positions
        score -= z * 10
        
        # Prefer positions against walls
        if x == 0 or x + dims[0] == self.dimensions[0]:
            score += 5
        if y == 0 or y + dims[1] == self.dimensions[1]:
            score += 5
            
        # Prefer positions that create flat surfaces for stacking
        if self._creates_stackable_surface(item, pos, dims):
            score += 8
            
        # Consider load bearing efficiency
        support_score = self._evaluate_support_efficiency(item, pos, dims)
        score += support_score * 3
        
        return score

    def _creates_stackable_surface(self, item, pos, dims):
        """Check if position creates good surface for stacking"""
        x, y, z = pos
        w, d, h = dims
        
        # Check if this creates a flat surface at same height as other items
        same_height_items = [i for i in self.items 
                           if abs((i.position[2] + i.dimensions[2]) - (z + h)) < 0.001]
        
        if same_height_items:
            return True
            
        return False

    def _check_stackability(self, item: Item, pos: Tuple[float, float, float]) -> bool:
        """Check if items below can support this item's weight"""
        x, y, z = pos
        w, d, h = item.dimensions
        
        # If item is on the ground, no stackability check needed
        if z == 0:
            return True
            
        # Find items directly below
        items_below = self._get_items_below(pos, (w, d))
        
        # If no items below but not on ground, item can't be supported
        if not items_below and z > 0:
            return False
            
        # Calculate total support capacity and total weight above
        total_support_capacity = 0
        area_supported = 0
        
        for below_item in items_below:
            # Calculate overlap area
            overlap = self._calculate_overlap_area(
                (x, y, w, d),
                (below_item.position[0], below_item.position[1],
                 below_item.dimensions[0], below_item.dimensions[1])
            )
            
            # Add support capacity proportional to overlap area
            area_ratio = overlap / (w * d)
            total_support_capacity += below_item.stackable * area_ratio
            area_supported += overlap
            
        # Check if total support capacity is sufficient
        total_area = w * d
        
        # Ensure minimum support area (30% of base)
        if area_supported / total_area < 0.3:
            return False
            
        return total_support_capacity >= item.weight

    def create_interactive_visualization(self):
        # Modify to use self instead of container parameter
        fig = sp.make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'scene', 'rowspan': 2}, {'type': 'table'}],
                [None, {'type': 'table'}]
            ],
            column_widths=[0.7, 0.3],
            row_heights=[0.6, 0.4],
            subplot_titles=('3D Container View', 'Unpacked Items', 'Available Spaces')
        )
        
        # Add container and items to 3D view
        self.add_container_boundaries(fig)
        self.add_items_with_bundles(fig)
        self.add_center_of_gravity(fig)
        
        # Add unpacked items table
        self.add_unpacked_table(fig)
        
        # Add remaining space table
        empty_spaces_df = pd.DataFrame([
            {
                'Location': f"({x:.2f}, {y:.2f}, {z:.2f})",
                'Dimensions': f"{w:.2f}m × {d:.2f}m × {h:.2f}m",
                'Volume': f"{w*d*h:.2f}m³"
            }
            for x, y, z, w, d, h in self.unused_spaces
        ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Location', 'Dimensions', 'Volume'],
                          fill=dict(color='lightgray'),
                          align='left'),
                cells=dict(values=[empty_spaces_df[col] for col in empty_spaces_df.columns],
                          align='left')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,  # Increase height for better visibility
            title=dict(
                text=(f'3D Container Loading<br>'
                      f'Volume Utilization: {self.volume_utilization:.1f}%<br>'
                      f'Remaining Volume: {self.remaining_volume:.2f}m³<br>'
                      f'Items Packed: {len(self.items)}/{len(self.items) + len(self.unpacked_reasons)}'),
                x=0.5,
                y=0.95
            ),
            showlegend=True
        )
        
        return fig

    def add_container_boundaries(self, fig):
        """Add container walls and edges to the visualization"""
        x, y, z = self.dimensions
        
        # Add container walls with better visibility
        faces = [
            # Bottom face
            go.Mesh3d(
                x=[0, x, x, 0],
                y=[0, 0, y, y],
                z=[0, 0, 0, 0],
                color='lightblue',
                opacity=0.3,
                showscale=False,
                name='Container Floor'
            ),
            # Back wall
            go.Mesh3d(
                x=[0, x, x, 0],
                y=[y, y, y, y],
                z=[0, 0, z, z],
                color='lightblue',
                opacity=0.3,
                showscale=False,
                name='Container Back'
            ),
            # Left wall
            go.Mesh3d(
                x=[0, 0, 0, 0],
                y=[0, y, y, 0],
                z=[0, 0, z, z],
                color='lightblue',
                opacity=0.3,
                showscale=False,
                name='Container Left'
            )
        ]
        
        # Add container edges for better definition
        edges = [
            # Floor edges
            go.Scatter3d(x=[0, x], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=3)),
            go.Scatter3d(x=[0, 0], y=[0, y], z=[0, 0], mode='lines', line=dict(color='black', width=3)),
            go.Scatter3d(x=[0, x], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=3)),
            go.Scatter3d(x=[0, 0], y=[0, y], z=[0, 0], mode='lines', line=dict(color='black', width=3)),
            go.Scatter3d(x=[x, x], y=[0, y], z=[0, 0], mode='lines', line=dict(color='black', width=3)),
            go.Scatter3d(x=[0, x], y=[y, y], z=[0, 0], mode='lines', line=dict(color='black', width=3)),
            # Vertical edges
            go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, z], mode='lines', line=dict(color='black', width=3)),
            go.Scatter3d(x=[x, x], y=[0, 0], z=[0, z], mode='lines', line=dict(color='black', width=3)),
            go.Scatter3d(x=[x, x], y=[y, y], z=[0, z], mode='lines', line=dict(color='black', width=3)),
            go.Scatter3d(x=[0, 0], y=[y, y], z=[0, z], mode='lines', line=dict(color='black', width=3)),
            # Top edges
            go.Scatter3d(x=[0, x], y=[0, 0], z=[z, z], mode='lines', line=dict(color='black', width=3)),
            go.Scatter3d(x=[0, 0], y=[0, y], z=[z, z], mode='lines', line=dict(color='black', width=3)),
            go.Scatter3d(x=[x, x], y=[0, y], z=[z, z], mode='lines', line=dict(color='black', width=3)),
            go.Scatter3d(x=[0, x], y=[y, y], z=[z, z], mode='lines', line=dict(color='black', width=3))
        ]
        
        # Add grid lines for better space perception
        grid_lines = []
        grid_spacing = min(x, y, z) / 10  # Create grid with 10 divisions
        
        # Add floor grid
        for i in np.arange(0, x + grid_spacing, grid_spacing):
            grid_lines.append(go.Scatter3d(x=[i, i], y=[0, y], z=[0, 0], 
                                         mode='lines', line=dict(color='gray', width=1)))
        for i in np.arange(0, y + grid_spacing, grid_spacing):
            grid_lines.append(go.Scatter3d(x=[0, x], y=[i, i], z=[0, 0], 
                                         mode='lines', line=dict(color='gray', width=1)))

        # Add all elements to the figure
        for face in faces:
            fig.add_trace(face)
        for edge in edges:
            fig.add_trace(edge)
        for grid in grid_lines:
            fig.add_trace(grid)

        # Update scene layout for better visualization
        fig.update_scenes(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data',
            xaxis=dict(range=[-x*0.1, x*1.1]),
            yaxis=dict(range=[-y*0.1, y*1.1]),
            zaxis=dict(range=[-z*0.1, z*1.1])
        )

    def add_item_to_plot(self, fig, item):
        x, y, z = item.position
        l, w, h = item.dimensions
        
        # Define vertices
        vertices = [
            [x, y, z], [x+l, y, z], [x+l, y+w, z], [x, y+w, z],
            [x, y, z+h], [x+l, y, z+h], [x+l, y+w, z+h], [x, y+w, z+h]
        ]
        
        # Define faces using vertices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # left
        ]

        # Add faces
        for face in faces:
            fig.add_trace(go.Mesh3d(
                x=[v[0] for v in face],
                y=[v[1] for v in face],
                z=[v[2] for v in face],
                i=[0, 0, 0, 0],
                j=[1, 2, 3, 1],
                k=[2, 3, 1, 2],
                color=item.color,
                opacity=0.7,
                hovertext=f"""
                Item: {item.name}<br>
                Position: ({x:.2f}, {y:.2f}, {z:.2f})<br>
                Dimensions: {l:.2f}m × {w:.2f}m × {h:.2f}m<br>
                Weight: {item.weight}kg<br>
                Quantity: {item.quantity}<br>
                Fragility: {item.fragility}
                """,
                hoverinfo='text',
                showscale=False,
                name=item.name
            ))
        
        # Add edges for each box
        edges = [
            # Bottom face
            ([x, x+l], [y, y], [z, z]), ([x+l, x+l], [y, y+w], [z, z]),
            ([x+l, x], [y+w, y+w], [z, z]), ([x, x], [y+w, y], [z, z]),
            # Top face
            ([x, x+l], [y, y], [z+h, z+h]), ([x+l, x+l], [y, y+w], [z+h, z+h]),
            ([x+l, x], [y+w, y+w], [z+h, z+h]), ([x, x], [y+w, y], [z+h, z+h]),
            # Vertical edges
            ([x, x], [y, y], [z, z+h]), ([x+l, x+l], [y, y], [z, z+h]),
            ([x+l, x+l], [y+w, y+w], [z, z+h]), ([x, x], [y+w, y+w], [z, z+h])
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ))

    def add_items_with_bundles(self, fig):
        """Add items and their bundle subdivisions to the visualization"""
        for item in self.items:
            if not item.position:
                continue
            self.add_item_to_plot(fig, item)
            if item.bundle == 'YES' and item.quantity > 1:
                self.add_bundle_subdivisions(fig, item)

    def add_bundle_subdivisions(self, fig, item):
        """Add visual subdivisions for bundled items"""
        x, y, z = item.position
        orig_l, orig_w, orig_h = item.original_dims
        qty = item.quantity
        
        # Calculate subdivision dimensions
        nx = int(item.dimensions[0] / orig_l)
        ny = int(item.dimensions[1] / orig_w)
        nz = int(item.dimensions[2] / orig_h)
        
        # Add inner edges for subdivisions
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    if i * j * k < qty:  # Only add subdivisions up to quantity
                        self.add_subdivision_edges(fig, 
                                           (x + i * orig_l, y + j * orig_w, z + k * orig_h),
                                           (orig_l, orig_w, orig_h),
                                           item.color)

    def add_center_of_gravity(self, fig):
        """Add center of gravity indicator to the visualization"""
        x, y, z = self.center_of_gravity
        
        # Add sphere at COG
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(
                size=10,
                symbol='circle',
                color='red',
                line=dict(color='black', width=2)
            ),
            name='Center of Gravity',
            hovertext=f'Center of Gravity<br>x: {x:.2f}, y: {y:.2f}, z: {z:.2f}',
            hoverinfo='text'
        ))
        
        # Add crosshairs
        self.add_cog_crosshairs(fig)

    def add_unpacked_table(self, fig):
        """Add table showing unpacked items and reasons"""
        if self.unpacked_reasons:
            df = pd.DataFrame([
                {
                    'Item': name,
                    'Reason': reason,
                    'Dimensions': f"{item.dimensions[0]}x{item.dimensions[1]}x{item.dimensions[2]}",
                    'Weight': item.weight
                }
                for name, (reason, item) in self.unpacked_reasons.items()
            ])
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=list(df.columns),
                        align='left'
                    ),
                    cells=dict(
                        values=[df[col] for col in df.columns],
                        align='left'
                    )
                ),
                row=1, col=2
            )

    def add_weight_distribution(self, fig):
        # Add weight distribution bars at the bottom
        sections = len(self.weight_distribution)
        if sections > 0:
            max_weight = max(self.weight_distribution.values())
            section_width = self.dimensions[0] / 3
            
            for section, weight in self.weight_distribution.items():
                height = (weight / max_weight) * (self.dimensions[2] * 0.1)
                x = section * section_width
                
                fig.add_trace(go.Mesh3d(
                    x=[x, x + section_width, x + section_width, x],
                    y=[0, 0, self.dimensions[1] * 0.1, self.dimensions[1] * 0.1],
                    z=[0, 0, 0, 0],
                    opacity=0.5,
                    color='red' if weight/max_weight > 0.4 else 'green',
                    name=f'Weight Section {section+1}: {weight:.0f}kg'
                ))

    def add_subdivision_edges(self, fig, pos, dims, color):
        """Add edges for bundle subdivisions"""
        x, y, z = pos
        l, w, h = dims
        
        # Define edges with thinner lines
        edges = [
            # Bottom face
            ([x, x+l], [y, y], [z, z]), ([x+l, x+l], [y, y+w], [z, z]),
            ([x+l, x], [y+w, y+w], [z, z]), ([x, x], [y+w, y], [z, z]),
            # Top face
            ([x, x+l], [y, y], [z+h, z+h]), ([x+l, x+l], [y, y+w], [z+h, z+h]),
            ([x+l, x], [y+w, y+w], [z+h, z+h]), ([x, x], [y+w, y], [z+h, z+h]),
            # Vertical edges
            ([x, x], [y, y], [z, z+h]), ([x+l, x+l], [y, y], [z, z+h]),
            ([x+l, x+l], [y+w, y+w], [z, z+h]), ([x, x], [y+w, y+w], [z, z+h])
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color=color, width=0.5, dash='dot'),
                showlegend=False
            ))

    def add_cog_crosshairs(self, fig):
        """Add crosshair lines through center of gravity"""
        x, y, z = self.center_of_gravity
        dims = self.dimensions
        
        # Add crosshair lines
        lines = [
            # Vertical line
            ([x, x], [y, y], [0, dims[2]]),
            # Width line
            ([x, x], [0, dims[1]], [z, z]),
            # Length line
            ([0, dims[0]], [y, y], [z, z])
        ]
        
        for line in lines:
            fig.add_trace(go.Scatter3d(
                x=line[0], y=line[1], z=line[2],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False
            ))

    def _calculate_weight_balance_score(self) -> float:
        """Calculate overall weight balance score"""
        if not self.weight_distribution:
            return 0.0
            
        weights = list(self.weight_distribution.values())
        total_weight = sum(weights)
        if total_weight == 0:
            return 1.0
            
        # Calculate variance in weight distribution
        mean_weight = total_weight / len(weights)
        variance = sum((w - mean_weight) ** 2 for w in weights) / len(weights)
        
        # Return normalized score (0-1, higher is better)
        return 1.0 / (1.0 + variance / 1000)

    def _calculate_interlocking_score(self) -> float:
        """Calculate overall interlocking score"""
        if not self.items:
            return 0.0
            
        total_contacts = 0
        for item in self.items:
            for other in self.items:
                if item != other and self._has_surface_contact(item.position, item.dimensions, other):
                    total_contacts += 1
                    
        # Return normalized score (0-1, higher is better)
        max_possible_contacts = len(self.items) * 6  # Each item can touch 6 sides
        return total_contacts / max_possible_contacts

    def _has_surface_contact(self, pos1, dims1, item2):
        """Check if two items have surface contact"""
        x1, y1, z1 = pos1
        l1, w1, h1 = dims1
        x2, y2, z2 = item2.position
        l2, w2, h2 = item2.dimensions
        
        # Check for surface contact on each face
        faces_contact = [
            # Bottom face contact
            abs(z1 - (z2 + h2)) < 0.001 and self._check_overlap_2d(
                (x1, y1, l1, w1), (x2, y2, l2, w2)),
            # Top face contact
            abs((z1 + h1) - z2) < 0.001 and self._check_overlap_2d(
                (x1, y1, l1, w1), (x2, y2, l2, w2)),
            # Front/back face contacts
            (abs(y1 - (y2 + w2)) < 0.001 or abs((y1 + w1) - y2) < 0.001) and
            self._check_overlap_2d((x1, z1, l1, h1), (x2, z2, l2, h2)),
            # Left/right face contacts
            (abs(x1 - (x2 + l2)) < 0.001 or abs((x1 + l1) - x2) < 0.001) and
            self._check_overlap_2d((y1, z1, w1, h1), (y2, z2, w2, h2))
        ]
        
        return any(faces_contact)

    # Create a Dash app for interactive visualization
    def create_interactive_app(self):
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.Button('Generate Alternative Arrangement', id='rearrange-button'),
            dcc.Graph(id='container-view', figure=self.create_interactive_visualization()),
            html.Div([
                html.H3('Packing Statistics'),
                html.P(id='stats-display')
            ])
        ])
        
        @app.callback(
            [Output('container-view', 'figure'),
             Output('stats-display', 'children')],
            [Input('rearrange-button', 'n_clicks')]
        )
        def update_arrangement(n_clicks):
            if n_clicks:
                self.generate_alternative_arrangement()
            
            fig = self.create_interactive_visualization()
            
            stats = f"""
            Volume Utilization: {self.volume_utilization:.1f}%
            Items Packed: {len(self.items)}
            Remaining Volume: {self.remaining_volume:.2f}m³
            """
            
            return fig, stats
        
        return app

    def generate_packing_report(self, filename=None):
        """Generate a detailed packing report in both human-readable and machine-readable formats"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"packing_report_{timestamp}"

        # Create report data structure
        report_data = {
            "container": {
                "dimensions": self.dimensions,
                "volume": self.total_volume,
                "volume_utilization": self.volume_utilization,
                "remaining_volume": self.remaining_volume,
                "center_of_gravity": self.center_of_gravity,
                "weight_distribution": self.weight_distribution,
                "total_weight": self.total_weight
            },
            "packed_items": [
                {
                    "name": item.name,
                    "position": item.position,
                    "dimensions": item.dimensions,
                    "original_dims": item.original_dims,
                    "weight": item.weight,
                    "quantity": item.quantity,
                    "fragility": item.fragility,
                    "stackable": item.stackable,  # Changed from load_bear
                    "bundle": item.bundle
                }
                for item in self.items
            ],
            "unpacked_items": [
                {
                    "name": name,
                    "reason": reason,
                    "dimensions": item.dimensions,
                    "weight": item.weight
                }
                for name, (reason, item) in self.unpacked_reasons.items()
            ],
            "unused_spaces": [
                {
                    "position": (x, y, z),
                    "dimensions": (w, d, h),
                    "volume": w * d * h
                }
                for x, y, z, w, d, h in self.unused_spaces
            ],
            "metrics": {
                "weight_balance_score": self._calculate_weight_balance_score(),
                "interlocking_score": self._calculate_interlocking_score(),
                "total_items": len(self.items),
                "unpacked_items": len(self.unpacked_reasons)
            }
        }

        # Save machine-readable JSON report
        with open(f"{filename}.json", 'w') as f:
            json.dump(report_data, f, indent=4)

        # Generate human-readable report
        with open(f"{filename}.txt", 'w') as f:
            f.write("=== Container Loading Report ===\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Container Specifications:\n")
            f.write(f"Dimensions (L×W×H): {self.dimensions[0]}m × {self.dimensions[1]}m × {self.dimensions[2]}m\n")
            f.write(f"Total Volume: {self.total_volume:.2f}m³\n")
            f.write(f"Volume Utilization: {self.volume_utilization:.1f}%\n")
            f.write(f"Remaining Volume: {self.remaining_volume:.2f}m³\n")
            f.write(f"Total Weight: {self.total_weight:.2f}kg\n")
            f.write(f"Center of Gravity: ({self.center_of_gravity[0]:.2f}, {self.center_of_gravity[1]:.2f}, {self.center_of_gravity[2]:.2f})\n\n")
            
            f.write("Packed Items:\n")
            for item in sorted(self.items, key=lambda x: x.position[2]):
                f.write(f"- {item.name}:\n")
                f.write(f"  Position: ({item.position[0]:.2f}, {item.position[1]:.2f}, {item.position[2]:.2f})\n")
                f.write(f"  Dimensions: {item.dimensions[0]:.2f}m × {item.dimensions[1]:.2f}m × {item.dimensions[2]:.2f}m\n")
                f.write(f"  Weight: {item.weight}kg\n")
                
            f.write("\nUnpacked Items:\n")
            for name, (reason, item) in self.unpacked_reasons.items():
                f.write(f"- {name}: {reason}\n")
                
            f.write("\nUnused Spaces:\n")
            for x, y, z, w, d, h in sorted(self.unused_spaces, key=lambda s: s[2]):
                f.write(f"- Position: ({x:.2f}, {y:.2f}, {z:.2f})\n")
                f.write(f"  Size: {w:.2f}m × {d:.2f}m × {h:.2f}m\n")
                f.write(f"  Volume: {w*d*h:.2f}m³\n")
                
            f.write("\nMetrics:\n")
            f.write(f"Weight Balance Score: {self._calculate_weight_balance_score():.2f}\n")
            f.write(f"Interlocking Score: {self._calculate_interlocking_score():.2f}\n")
            
        return report_data

    def main():
        print("=== 3D Container Loading Optimizer with Genetic Algorithm ===")
        
        try:
            # Get container configuration
            container_dims = get_transport_config()
            
            # Create enhanced container
            container = EnhancedContainer(container_dims)
            
            # Read data
            df = pd.read_csv('optmised_container_data.csv')
            
            # Create items from CSV data with Stackable instead of LoadBear
            items = []
            for _, row in df.iterrows():
                try:
                    item = Item(
                        name=row['Name'],
                        length=row['Length'],
                        width=row['Width'],
                        height=row['Height'],
                        weight=row['Weight'],
                        quantity=row['Quantity'],
                        fragility=row['Fragility'],
                        stackable=row['Stackable'],  # Changed from load_bear to stackable
                        boxing_type=row['BoxingType'],
                        bundle=row['Bundle']
                    )
                    items.append(item)
                except KeyError as e:
                    print(f"Error reading column: {e}")
                    print("Available columns:", list(row.index))
                    return
                except Exception as e:
                    print(f"Error processing item: {e}")
                    continue
            
            if not items:
                print("No items were loaded successfully")
                return
                
            # Replace direct packing with genetic algorithm optimization
            container = optimize_packing_with_genetic_algorithm(
                items=items,
                container_dims=container_dims,
                population_size=50,
                generations=100
            )
            
            # Calculate utilization
            container.volume_utilization = (sum(
                item.dimensions[0] * item.dimensions[1] * item.dimensions[2] 
                for item in container.items
            ) / (container_dims[0] * container_dims[1] * container_dims[2])) * 100
            
            # Print packing results
            print("\n=== Packing Results ===")
            print(f"Container dimensions: {container_dims[0]}m x {container_dims[1]}m x {container_dims[2]}m")
            print(f"Total items packed: {len(container.items)}")
            print(f"Volume utilization: {container.volume_utilization:.1f}%")
            print("\nPacked items:")
            for item in container.items:
                print(f"- {item.name} at position {item.position}")
                print(f"- {item.name} at position {item.position}")
            print("\nUnpacked items:")
            unpacked = [item.name for item in items if item not in container.items]
            for item in unpacked:
                print(f"- {item}")
            
            # Visualize results
            fig = container.create_interactive_visualization()
            fig.show()
            
            # Add detailed packing analysis
            print("\n=== Packing Analysis ===")
            print(f"Center of Gravity: ({container.center_of_gravity[0]:.2f}, "
                  f"{container.center_of_gravity[1]:.2f}, {container.center_of_gravity[2]:.2f})")
            print(f"Weight Distribution Balance: {container._calculate_weight_balance_score():.2f}")
            print(f"Interlocking Score: {container._calculate_interlocking_score():.2f}")
            
            # Create and run interactive app instead of just showing the figure
            app = container.create_interactive_app()
            app.run_server(debug=True)
            
            # Generate packing report
            report_data = container.generate_packing_report()
            print(f"\nReport generated: packing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json/txt")
            
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

    if __name__ == "__main__":    main()

    def _find_best_position_advanced(self, item, rotations):
        """Enhanced position finding with advanced stability and space utilization"""
        best_score = float('-inf')
        best_pos = None
        best_rot = None
        best_space = None

        for rotation in rotations:
            for space in self.spaces:
                if not space.can_fit_item(rotation):
                    continue

                # Try positions optimizing for stability and space utilization
                positions = [
                    (space.x, space.y, space.z),  # Default position
                    (space.x + space.width - rotation[0], space.y, space.z),  # Right aligned
                    (space.x, space.y + space.depth - rotation[1], space.z),  # Back aligned
                    (space.x + space.width - rotation[0], 
                     space.y + space.depth - rotation[1], space.z)  # Corner aligned
                ]

                for pos in positions:
                    if not self._is_valid_placement(item, pos, rotation):
                        continue

                    # Calculate comprehensive placement score
                    score = (
                        self._evaluate_stability(item, pos, rotation) * 0.4 +
                        self._evaluate_space_utilization(pos, rotation, space) * 0.3 +
                        self._evaluate_weight_distribution(item, pos) * 0.2 +
                        self._evaluate_accessibility(pos, rotation) * 0.1
                    )

                    if score > best_score:
                        best_score = score
                        best_pos = pos
                        best_rot = rotation
                        best_space = space

        return best_pos, best_rot, best_space

    def _evaluate_stability(self, item, pos, dims):
        """Evaluate stability of item placement"""
        score = 0
        x, y, z = pos
        w, d, h = dims

        # Base support score
        support_score = self._calculate_support_score(item, pos, dims)
        score += support_score * 5

        # Center of gravity impact
        cog_score = self._evaluate_cog_impact(item, pos)
        score += cog_score * 3

        # Corner and edge support
        corner_support = self._calculate_corner_support(pos, dims)
        score += corner_support * 2

        # Wall contact bonus
        wall_contact = self._calculate_wall_contact(pos, dims)
        score += wall_contact

        return score

    def _evaluate_space_utilization(self, pos, dims, space):
        """Evaluate how well the placement utilizes available space"""
        # Calculate volume utilization
        item_volume = dims[0] * dims[1] * dims[2]
        space_volume = space.width * space.height * space.depth
        volume_ratio = item_volume / space_volume

        # Calculate remaining space usability
        remaining_spaces = self._simulate_space_splitting(pos, dims, space)
        usable_volume = sum(s[3] * s[4] * s[5] for s in remaining_spaces)
        usability_score = usable_volume / (space_volume - item_volume) if space_volume > item_volume else 0

        return (volume_ratio * 0.6 + usability_score * 0.4)

    def _simulate_space_splitting(self, pos, dims, space):
        """Simulate how space would be split after placement"""
        x, y, z = pos
        w, d, h = dims
        remaining_spaces = []

        # Above space
        if space.height > h:
            remaining_spaces.append((x, y, z + h, w, d, space.height - h))

        # Right space
        if space.width > w:
            remaining_spaces.append((x + w, y, z, space.width - w, d, h))

        # Front space
        if space.depth > d:
            remaining_spaces.append((x, y + d, z, w, space.depth - d, h))

        return remaining_spaces

    def _evaluate_weight_distribution(self, item, pos):
        """Evaluate impact on weight distribution"""
        x = pos[0]
        section_width = self.dimensions[0] / 3
        current_section = int(x / section_width)
        
        # Calculate current weight distribution
        section_weights = [0, 0, 0]
        for existing_item in self.items:
            item_section = int(existing_item.position[0] / section_width)
            section_weights[item_section] += existing_item.weight
        
        # Add new item weight
        test_weights = section_weights.copy()
        test_weights[current_section] += item.weight
        
        # Calculate balance score
        total_weight = sum(test_weights)
        if total_weight == 0:
            return 1.0
            
        ideal_weight = total_weight / 3
        max_deviation = max(abs(w - ideal_weight) for w in test_weights)
        
        return 1.0 - (max_deviation / total_weight)

    def _evaluate_accessibility(self, pos, dims):
        """Evaluate how accessible the item would be for loading/unloading"""
        x, y, z = pos
        w, d, h = dims
        
        # Calculate distance from container entrance
        distance_score = 1.0 - (y / self.dimensions[1])
        
        # Evaluate clearance above
        clearance_score = self._calculate_clearance(pos, dims)
        
        return (distance_score * 0.7 + clearance_score * 0.3)

    def _calculate_clearance(self, pos, dims):
        """Calculate vertical and horizontal clearance around position"""
        x, y, z = pos
        w, d, h = dims
        clearance = 0
        
        # Check vertical clearance
        max_height = self.dimensions[2]
        vertical_space = (max_height - (z + h)) / max_height
        clearance += vertical_space
        
        # Check horizontal clearance
        horizontal_space = min(
            x / self.dimensions[0],
            (self.dimensions[0] - (x + w)) / self.dimensions[0],
            y / self.dimensions[1],
            (self.dimensions[1] - (y + d)) / self.dimensions[1]
        )
        clearance += horizontal_space
        
        return clearance / 2

    def _calculate_corner_support(self, pos, dims):
        """Calculate support at corners of the item"""
        x, y, z = pos
        w, d, h = dims
        corners = [
            (x, y), (x + w, y),
            (x, y + d), (x + w, y + d)
        ]
        
        supported_corners = 0
        for corner_x, corner_y in corners:
            if self._is_corner_supported(corner_x, corner_y, z):
                supported_corners += 1
                
        return supported_corners / 4

    def _is_corner_supported(self, x, y, z):
        """Check if a specific corner has support"""
        if z == 0:  # On container floor
            return True
            
        tolerance = 0.001
        for item in self.items:
            if abs(item.position[2] + item.dimensions[2] - z) < tolerance:
                if (item.position[0] <= x <= item.position[0] + item.dimensions[0] and
                    item.position[1] <= y <= item.position[1] + item.dimensions[1]):
                    return True
                    
        return False

class PackingGenome:
    def __init__(self, items, mutation_rate=0.1):
        self.item_sequence = items.copy()
        self.rotation_flags = [random.randint(0, 5) for _ in items]  # 6 possible rotations
        self.mutation_rate = mutation_rate
        self.fitness = 0.0

    def mutate(self):
        """Apply mutation operators"""
        for i in range(len(self.item_sequence)):
            if random.random() < self.mutation_rate:
                # Sequence mutation
                j = random.randint(0, len(self.item_sequence) - 1)
                self.item_sequence[i], self.item_sequence[j] = self.item_sequence[j], self.item_sequence[i]
                # Rotation mutation
                self.rotation_flags[i] = random.randint(0, 5)

class GeneticPacker:
    def __init__(self, container_dims, population_size=50, generations=100):
        self.container_dims = container_dims
        self.population_size = population_size
        self.generations = generations
        self.best_solution = None
        self.best_fitness = 0.0

    def optimize(self, items):
        """Run genetic algorithm optimization"""
        # Initialize population
        population = [PackingGenome(items) for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            # Evaluate fitness for each genome
            for genome in population:
                genome.fitness = self._evaluate_fitness(genome)
                
                # Update best solution
                if genome.fitness > self.best_fitness:
                    self.best_fitness = genome.fitness
                    self.best_solution = genome
            
            # Selection and reproduction
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                child = self._crossover(parent1, parent2)
                child.mutate()
                new_population.append(child)
                
            population = new_population
            
            # Print progress
            avg_fitness = sum(g.fitness for g in population) / len(population)
            print(f"Generation {generation + 1}: Best Fitness = {self.best_fitness:.4f}, Avg Fitness = {avg_fitness:.4f}")
        
        return self.best_solution

    def _evaluate_fitness(self, genome):
        """Evaluate packing fitness"""
        container = EnhancedContainer(self.container_dims)
        
        # Try to pack items in the sequence specified by genome
        for item, rotation_flag in zip(genome.item_sequence, genome.rotation_flags):
            # Apply rotation based on flag
            original_dims = item.dimensions
            item.dimensions = self._get_rotation(original_dims, rotation_flag)
            
            # Try to pack with current rotation
            rotations = [item.dimensions]  # Only try specified rotation
            best_pos = None
            best_rot = None
            best_space = None
            
            for space in container.spaces:
                if space.can_fit_item(item.dimensions):
                    pos = (space.x, space.y, space.z)
                    if container._is_valid_placement(item, pos, item.dimensions):
                        best_pos = pos
                        best_rot = item.dimensions
                        best_space = space
                        break
            
            if best_pos:
                item.position = best_pos
                item.dimensions = best_rot
                container.items.append(item)
                container._update_spaces(best_pos, best_rot, best_space)
            
            # Restore original dimensions
            item.dimensions = original_dims
        
        # Calculate fitness components
        volume_utilization = (sum(
            item.dimensions[0] * item.dimensions[1] * item.dimensions[2] 
            for item in container.items
        ) / (container.dimensions[0] * container.dimensions[1] * container.dimensions[2]))
        
        stability_score = sum(
            container._calculate_stability_score(item, item.position, item.dimensions)
            for item in container.items
        ) / len(container.items) if container.items else 0
        
        weight_balance = container._calculate_weight_balance_score()
        items_packed_ratio = len(container.items) / len(genome.item_sequence)
        
        # Combine fitness components
        fitness = (
            volume_utilization * 0.4 +
            stability_score * 0.3 +
            weight_balance * 0.2 +
            items_packed_ratio * 0.1
        )
        
        return fitness

    def _tournament_select(self, population, tournament_size=3):
        """Tournament selection"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1, parent2):
        """Order crossover (OX) for sequence, uniform crossover for rotations"""
        # OX crossover for item sequence
        size = len(parent1.item_sequence)
        start, end = sorted(random.sample(range(size), 2))
        
        # Create child sequence using OX
        child_sequence = [None] * size
        child_sequence[start:end] = parent1.item_sequence[start:end]
        
        remaining = [item for item in parent2.item_sequence 
                    if item not in child_sequence[start:end]]
        
        j = 0
        for i in range(size):
            if child_sequence[i] is None:
                child_sequence[i] = remaining[j]
                j += 1
        
        # Uniform crossover for rotations
        child_rotations = [
            parent1.rotation_flags[i] if random.random() < 0.5 
            else parent2.rotation_flags[i]
            for i in range(size)
        ]
        
        child = PackingGenome(child_sequence)
        child.rotation_flags = child_rotations
        return child

    def _get_rotation(self, dims, flag):
        """Get dimensions after rotation based on flag"""
        l, w, h = dims
        rotations = [
            (l, w, h), (l, h, w), (w, l, h),
            (w, h, l), (h, l, w), (h, w, l)
        ]
        return rotations[flag]

def optimize_packing_with_genetic_algorithm(items, container_dims, 
                                         population_size=50, generations=100):
    """Main function to optimize packing using genetic algorithm"""
    genetic_packer = GeneticPacker(container_dims, population_size, generations)
    best_genome = genetic_packer.optimize(items)
    
    # Create final container with best solution
    container = EnhancedContainer(container_dims)
    
    # Pack items according to best genome
    for item, rotation_flag in zip(best_genome.item_sequence, best_genome.rotation_flags):
        item.dimensions = genetic_packer._get_rotation(item.original_dims, rotation_flag)
        container.pack_items([item])
    
    return container

def can_interlock(item1, item2) -> bool:
    """Check if two items can potentially interlock"""
    if not (item1 and item2):
        return False
        
    try:
        dims1 = item1.dimensions
        dims2 = item2.dimensions
        
        # Check if any dimension pairs are similar (within 10%)
        for d1 in dims1:
            for d2 in dims2:
                if abs(d1 - d2) / max(d1, d2) < 0.1:
                    return True
                    
        # Check if items can stack
        if item1.stackable == 'YES' and item2.stackable == 'YES':
            return True
            
        return False
        
    except Exception:
        return False



