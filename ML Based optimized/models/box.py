import numpy as np

class Box:
    def __init__(self, box_id, category, dimensions, weight, fragility, 
                 load_bearing, quantity, shape, packaging_type, temp_req, contents):
        # Parse dimensions string (format: "LxWxH")
        l, w, h = map(int, dimensions.split('x'))
        self.dimensions = np.array([l/100, w/100, h/100])  # Convert cm to meters
        
        self.box_id = box_id
        self.weight = weight
        self.load_bearing = load_bearing
        self.quantity = quantity
        self.properties = {
            'fragility': fragility,
            'category': category,
            'temp_req': temp_req == 'Cool',  # Convert to boolean
            'stackable': True if load_bearing > 0 else False,
            'shape': shape,
            'packaging': packaging_type,
            'contents': contents
        }
        self.norm_dimensions = self.dimensions.copy()  # Initialize normalized dimensions
    
    def normalize_features(self, container_dims):
        self.norm_dimensions = self.dimensions / container_dims
        return self.norm_dimensions
        
    def get_feature_vector(self):
        """Updated to access fragility from properties dictionary"""
        features = [
            self.dimensions[0],  # length
            self.dimensions[1],  # width
            self.dimensions[2],  # height
            self.weight,
            self.load_bearing,
            float(self.properties['fragility'] == "High"),
            float(self.properties['fragility'] == "Medium"),
            float(self.properties['fragility'] == "Low"),
            float(self.properties['shape'] == "Regular"),
            float(self.properties['shape'] == "Irregular"),
            float(self.properties['temp_req'] == "Ambient"),
            float(self.properties['temp_req'] == "Cool"),
            float(self.properties['temp_req'] == "Frozen")
        ]
        return np.array(features, dtype=np.float32)
