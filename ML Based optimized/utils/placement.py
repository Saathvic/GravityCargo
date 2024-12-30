import numpy as np

def get_possible_orientations(box, is_fragile=False):
    """Get possible orientations based on fragility"""
    l, w, h = box.dimensions
    
    if is_fragile:
        # Fragile boxes keep original orientation only
        return [(l, w, h)]
    else:
        # Non-fragile boxes can be rotated in all 6 possible ways
        return [
            (l, w, h), (l, h, w),
            (w, l, h), (w, h, l),
            (h, l, w), (h, w, l)
        ]

def check_load_bearing(placement, existing_placements, box):
    """Check if box can bear the weight of boxes above it"""
    pos_x, pos_y, pos_z = placement['position']
    dims = placement['dimensions']
    
    # Check boxes above this position
    for p in existing_placements:
        if (p['position'][0] >= pos_x and 
            p['position'][0] < pos_x + dims[0] and
            p['position'][2] >= pos_z and 
            p['position'][2] < pos_z + dims[2] and
            p['position'][1] > pos_y):
            
            # Calculate weight pressure
            if box.load_bearing < p['box'].weight:
                return False
    return True

def smart_placement_strategy(boxes, container):
    """Enhanced placement strategy with fragility and rotation handling"""
    placements = []
    current_layer_height = 0
    max_dims = container.dimensions
    
    # Sort boxes by fragility (less fragile first) and size (larger first)
    sorted_boxes = sorted(boxes, key=lambda x: (
        x.properties['fragility'] == 'High',
        x.properties['fragility'] == 'Medium',
        -np.prod(x.dimensions)  # Negative for descending order
    ))
    
    for box in sorted_boxes:
        placed = False
        is_fragile = box.properties['fragility'] in ['High', 'Medium']
        
        # Get possible orientations based on fragility
        orientations = get_possible_orientations(box, is_fragile)
        
        # Try each orientation
        for dims in orientations:
            if placed:
                break
                
            # Try to find a suitable position
            for y in range(0, int(max_dims[1] * 100), 10):  # Step by 10cm
                y = y / 100  # Convert back to meters
                if y + dims[1] > max_dims[1]:
                    continue
                    
                for z in range(0, int(max_dims[2] * 100), 10):
                    z = z / 100
                    if z + dims[2] > max_dims[2]:
                        continue
                        
                    for x in range(0, int(max_dims[0] * 100), 10):
                        x = x / 100
                        if x + dims[0] > max_dims[0]:
                            continue
                        
                        # Create potential placement
                        placement = {
                            'box': box,
                            'position': [x, y, z],
                            'dimensions': dims,
                            'orientation': orientations.index(dims)
                        }
                        
                        # Check if position is valid
                        if check_position_valid(placement, placements, max_dims):
                            # For non-fragile boxes, check load bearing capacity
                            if not is_fragile:
                                if check_load_bearing(placement, placements, box):
                                    placements.append(placement)
                                    placed = True
                                    break
                            else:
                                # Fragile boxes don't need load bearing check
                                placements.append(placement)
                                placed = True
                                break
                    
                    if placed:
                        break
                if placed:
                    break
    
    return placements

def check_position_valid(new_placement, existing_placements, container_dims):
    """Check if position is valid and doesn't overlap with existing boxes"""
    pos = new_placement['position']
    dims = new_placement['dimensions']
    
    # Check container boundaries
    if (pos[0] + dims[0] > container_dims[0] or
        pos[1] + dims[1] > container_dims[1] or
        pos[2] + dims[2] > container_dims[2]):
        return False
    
    # Check overlap with existing boxes
    for p in existing_placements:
        if boxes_overlap(new_placement, p):
            return False
    
    return True

def boxes_overlap(p1, p2):
    """Check if two box placements overlap"""
    return (
        p1['position'][0] < p2['position'][0] + p2['dimensions'][0] and
        p1['position'][0] + p1['dimensions'][0] > p2['position'][0] and
        p1['position'][1] < p2['position'][1] + p2['dimensions'][1] and
        p1['position'][1] + p1['dimensions'][1] > p2['position'][1] and
        p1['position'][2] < p2['position'][2] + p2['dimensions'][2] and
        p1['position'][2] + p1['dimensions'][2] > p2['position'][2]
    )
