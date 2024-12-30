from flask import Flask, jsonify, request, render_template
from flask_sock import Sock
import sys
from pathlib import Path
import json
import threading
from queue import Queue
import time

# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent))

from models.box import Box
from models.container import Container
from utils.preprocessing import normalize_dimensions
from utils.data_loader import load_boxes_from_excel
from utils.placement import smart_placement_strategy  # Use smart placement only
import torch
import numpy as np
from models.neural_net import PackingNetwork
from rl.packing_env import PackingEnvironment
from rl.train_rl import train_dqn

app = Flask(__name__)
sock = Sock(app)

# Initialize container and load boxes
try:
    container = Container()
    boxes = load_boxes_from_excel('Larger_Box_Dimensions.xlsx')
    print(f"Successfully loaded {len(boxes)} boxes")
except Exception as e:
    print(f"Error initializing: {str(e)}")
    boxes = []

# Initialize model with error handling
try:
    model = PackingNetwork()
    if Path('model_weights.pth').exists():
        state_dict = torch.load('model_weights.pth', weights_only=True)
        model.load_legacy_weights(state_dict)  # Use the new loading function
        model.eval()
        print("Model loaded successfully")
    else:
        print("No pre-trained weights found. Using untrained model.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = PackingNetwork()

# Add message queue for WebSocket
message_queue = Queue()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/pack', methods=['POST'])
def pack_items():
    data = request.get_json()
    return jsonify({"status": "success"})

@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    return jsonify({"status": "ready"})

@app.route('/api/boxes', methods=['GET'])
def get_boxes():
    return jsonify([{
        'id': box.box_id,
        'dimensions': box.dimensions.tolist(),
        'weight': box.weight,
        'load_bearing': box.load_bearing,
        'properties': box.properties,
        'volume': float(np.prod(box.dimensions))  # Calculate volume in m³
    } for box in boxes])

@app.route('/api/boxes/placement', methods=['GET'])
def get_box_placement():
    try:
        send_progress_update("Starting box placement...")
        placements = smart_placement_strategy(boxes, container)
        total = len(boxes)
        packed = len(placements)
        
        # Send completion message
        send_progress_update(f"Placement complete: {packed}/{total} boxes packed")
        
        return jsonify({
            'placements': [{
                'id': p['box'].box_id,
                'dimensions': p['dimensions'],
                'position': p['position'],
                'orientation': p['orientation'],
                'weight': p['box'].weight,
                'properties': p['box'].properties,
                'progress': f"{idx + 1}/{packed}"
            } for idx, p in enumerate(placements)],
            'statistics': {
                'total_boxes': total,
                'packed_boxes': packed,
                'packing_ratio': packed/total if total > 0 else 0,
                'fragile_boxes_top': sum(1 for p in placements 
                    if p['box'].properties['fragility'] == 'High' 
                    and p['position'][1] > container.dimensions[1]/2)
            }
        })
    except Exception as e:
        send_progress_update(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@sock.route('/ws')
def ws(sock):
    """Improved WebSocket handler with proper error handling and cleanup"""
    try:
        while True:
            # Check if there are messages to send
            try:
                message = message_queue.get(timeout=1.0)  # 1 second timeout
                sock.send(json.dumps(message))
            except Queue.Empty:
                # Send heartbeat to keep connection alive
                sock.send(json.dumps({'type': 'heartbeat'}))
            
            time.sleep(0.1)  # Prevent CPU overuse
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        try:
            sock.close()
        except:
            pass

def send_progress_update(message):
    """Helper function to send updates via WebSocket"""
    try:
        message_queue.put({
            'type': 'progress',
            'message': message
        })
    except Exception as e:
        print(f"Error sending progress update: {str(e)}")

@app.route('/api/predict_placement', methods=['POST'])
def predict_placement():
    try:
        data = request.get_json()
        box_features = torch.tensor(data['box_features'], dtype=torch.float32)
        container_state = torch.tensor(data['container_state'], dtype=torch.float32)
        
        with torch.no_grad():
            position, orientation = model(box_features, container_state)
        
        return jsonify({
            'position': position.numpy().tolist(),
            'orientation': orientation.numpy().tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rl/train', methods=['POST'])
def train_rl():
    # Example usage
    env = PackingEnvironment((32,32,32), boxes)  # Simplify dims
    trained_agent = train_dqn(env)
    return jsonify({"status": "RL training complete"})

if __name__ == '__main__':
    app.run(debug=True)
