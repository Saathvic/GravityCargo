import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_dimensions(dims, container_dims):
    return dims / container_dims

def create_training_pair(boxes, successful_placement):
    features = np.array([box.get_feature_vector() for box in boxes])
    return features, successful_placement

class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit_transform(self, features):
        return self.scaler.fit_transform(features)
        
    def transform(self, features):
        return self.scaler.transform(features)
