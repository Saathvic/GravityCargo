import numpy as np

class Container:
    # TEU (Twenty-foot Equivalent Unit) specifications
    EXTERNAL_DIMS = {
        'length': 6.06,  # meters
        'width': 2.44,   # meters
        'height': 2.59   # meters
    }
    
    INTERNAL_DIMS = {
        'length': 5.90,  # meters
        'width': 2.35,   # meters
        'height': 2.39   # meters
    }
    
    MAX_CAPACITY = 33.2  # cubic meters
    MAX_WEIGHT = 21770   # kg (20-foot container capacity)
    
    def __init__(self):
        self.length = self.INTERNAL_DIMS['length']
        self.width = self.INTERNAL_DIMS['width']
        self.height = self.INTERNAL_DIMS['height']
        self.max_weight = self.MAX_WEIGHT
        
        self.dimensions = np.array([self.length, self.width, self.height])
        self.space_matrix = np.zeros((
            int(self.length * 100),  # Convert to centimeters for grid
            int(self.width * 100),
            int(self.height * 100)
        ))
        self.current_weight = 0
        self.boxes_placed = []
        
    def get_normalized_dimensions(self):
        return self.dimensions / np.max(self.dimensions)
    
    def get_capacity(self):
        return self.MAX_CAPACITY
        
    def is_position_valid(self, position, box_dims):
        x, y, z = position
        l, w, h = box_dims
        
        if x + l > self.dimensions[0] or \
           y + w > self.dimensions[1] or \
           z + h > self.dimensions[2]:
            return False
        return True
