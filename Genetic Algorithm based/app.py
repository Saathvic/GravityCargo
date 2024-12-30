from flask import Flask, render_template, request, jsonify, send_file, session, url_for
from werkzeug.utils import secure_filename
import os
from GravicARgo_packing import *
import pandas as pd
import io
import numpy as np
import json
import plotly.graph_objects as go
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.secret_key = 'your_secret_key_here'  # Required for session
app.json_encoder = NumpyEncoder

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Add after the import section
ALLOWED_EXTENSIONS = {'csv'}
MIME_TYPES = {
    'json': 'application/json',
    'csv': 'text/csv',
    'html': 'text/html'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File is too large. Maximum size is 16MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Resource not found'
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'error': 'Internal server error occurred'
    }), 500

def cleanup_old_files():
    """Remove uploaded files older than 24 hours"""
    import time
    now = time.time()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.getmtime(filepath) < now - 86400:  # 24 hours
            try:
                os.remove(filepath)
            except OSError:
                pass

# Global container storage
class ContainerStorage:
    def __init__(self):
        self.current_container = None
        self.current_report = None

container_storage = ContainerStorage()

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/start')
def start():
    # Format transport modes and their supported containers
    formatted_modes = []
    for mode_id, (mode_name, containers) in TRANSPORT_MODES.items():
        container_details = []
        for container_name in containers:
            if container_name in CONTAINER_TYPES:
                dims = CONTAINER_TYPES[container_name]
                container_details.append({
                    'name': container_name,
                    'dimensions': dims,
                    'volume': dims[0] * dims[1] * dims[2]
                })
        
        formatted_modes.append({
            'id': mode_id,
            'name': mode_name,
            'containers': container_details
        })

    # Add default data for JavaScript
    default_data = {
        'transport_modes': formatted_modes,
        'container_types': CONTAINER_TYPES
    }

    return render_template('index.html', data=default_data)

# Modify the optimize route to include error handling and file validation
@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        cleanup_old_files()  # Cleanup old uploads
        
        # Validate file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV file'}), 400

        # Validate transport mode and container selection
        transport_mode = request.form.get('transport_mode')
        container_type = request.form.get('container_type')
        
        # Create container_info dictionary
        container_info = {
            'type': container_type if transport_mode != '5' else 'Custom',
            'transport_mode': TRANSPORT_MODES[transport_mode][0]  # Gets mode name like 'Sea Transport'
        }
        
        # Get container dimensions
        try:
            if transport_mode == '5':
                dimensions = (
                    float(request.form['length']),
                    float(request.form['width']),
                    float(request.form['height'])
                )
                container_info['dimensions'] = f"{dimensions[0]}m × {dimensions[1]}m × {dimensions[2]}m"
            else:
                dimensions = CONTAINER_TYPES[container_type]
                container_info['dimensions'] = f"{dimensions[0]}m × {dimensions[1]}m × {dimensions[2]}m"
                
        except (ValueError, KeyError):
            return jsonify({'error': 'Invalid container dimensions'}), 400

        # Handle file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
            
        # Save and process file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Create container and load items
        container = EnhancedContainer(dimensions)
        df = pd.read_csv(filepath)
        
        # Validate and standardize column names
        required_columns = ['Name', 'Length', 'Width', 'Height', 'Weight', 'Quantity', 'Fragility', 'BoxingType', 'Bundle']
        column_mapping = {
            'stackable': ['Stackable', 'LoadBearing', 'CanStack', 'Stack'],
            'fragility': ['Fragility', 'Fragile', 'FragilityLevel'],
            'boxing_type': ['BoxingType', 'PackagingType', 'Package'],
            'bundle': ['Bundle', 'IsBundled', 'Bundled']
        }
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'Missing required columns: {", ".join(missing_columns)}'}), 400
            
        # Find stackable column
        stackable_col = None
        for possible_name in column_mapping['stackable']:
            if possible_name in df.columns:
                stackable_col = possible_name
                break
        
        items = []
        warnings = []
        
        for _, row in df.iterrows():
            try:
                # Get stackable value with default
                stackable_value = 'YES' if stackable_col and row[stackable_col] in ['YES', 'Y', 'TRUE', '1'] else 'NO'
                
                # Validate bundle quantities
                if str(row['Bundle']).upper() in ['YES', 'Y', 'TRUE', '1']:
                    max_stack_height = dimensions[2] / float(row['Height'])
                    if row['Quantity'] > max_stack_height * 3:
                        warnings.append(
                            f"Warning: Bundle quantity ({row['Quantity']}) for {row['Name']} "
                            f"may be too large for container height {dimensions[2]}m. "
                            "Consider splitting into smaller bundles."
                        )
                
                # Create item with validated dimensions and defaults
                item = Item(
                    name=str(row['Name']),
                    length=min(float(row['Length']), dimensions[0]),
                    width=min(float(row['Width']), dimensions[1]),
                    height=min(float(row['Height']), dimensions[2]),
                    weight=float(row['Weight']),
                    quantity=int(row['Quantity']),
                    fragility=str(row['Fragility']).upper(),
                    stackable=stackable_value,
                    boxing_type=str(row['BoxingType']),
                    bundle=str(row['Bundle']).upper() in ['YES', 'Y', 'TRUE', '1']
                )
                items.append(item)
                app.logger.debug(f"Successfully processed item: {row['Name']}")
                
            except Exception as e:
                app.logger.error(f"Error processing item {row['Name']}: {str(e)}")
                warnings.append(f"Warning: Skipped item {row['Name']} due to error: {str(e)}")
                continue
        
        if not items:
            app.logger.error("No valid items found in CSV")
            app.logger.debug(f"CSV Preview:\n{df.head()}")
            return jsonify({
                'error': 'No valid items found in CSV',
                'details': 'Please ensure all required fields are present and valid',
                'preview': df.head().to_dict('records')
            }), 400
        
        # Pack items
        container.pack_items(items)
        
        # Store container and generate report
        container_storage.current_container = container
        
        # Convert numpy arrays to lists for JSON serialization
        report_data = {
            'container_dims': list(dimensions),
            'volume_utilization': float(container.volume_utilization),
            'items_packed': len(container.items),
            'total_items': len(items),
            'remaining_volume': float(container.remaining_volume),
            'center_of_gravity': [float(x) for x in container.center_of_gravity],
            'total_weight': float(container.total_weight)
        }
        
        container_storage.current_report = report_data
        
        # Create visualization with container info
        fig = create_interactive_visualization(container, container_info)
        
        return render_template('results.html',
                             plot=fig.to_html(),
                             container=container,
                             container_info=container_info,
                             report=report_data,
                             warnings=warnings)
                             
    except ValueError as e:
        return jsonify({'error': f'Invalid value in input: {str(e)}'}), 400
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'The uploaded CSV file is empty'}), 400
    except pd.errors.ParserError:
        return jsonify({'error': 'Unable to parse CSV file. Please check the format'}), 400
    except Exception as e:
        app.logger.error(f'Error during optimization: {str(e)}')
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/download_report')
def download_report():
    if container_storage.current_container is None:
        return jsonify({'error': 'No container data available'})
    
    try:
        container = container_storage.current_container
        report = generate_detailed_report(container)
        
        # Generate both JSON and HTML reports
        if request.args.get('format') == 'json':
            buffer = io.StringIO()
            json.dump(report, buffer, indent=4, cls=NumpyEncoder)
            buffer.seek(0)
            return send_file(
                buffer,
                as_attachment=True,
                download_name='packing_report.json',
                mimetype='application/json'
            )
        else:
            return render_template('downloadable_report.html',
                                report=report,
                                container=container)
    except Exception as e:
        return jsonify({'error': f'Error generating report: {str(e)}'})

@app.route('/alternative')
def generate_alternative():
    if container_storage.current_container is None:
        return jsonify({'error': 'No container data available'})
    
    container_storage.current_container.generate_alternative_arrangement()
    fig = create_interactive_visualization(container_storage.current_container)
    
    return jsonify({
        'plot': fig.to_html(),
        'stats': {
            'utilization': container_storage.current_container.volume_utilization,
            'items_packed': len(container_storage.current_container.items),
            'remaining_volume': container_storage.current_container.remaining_volume
        }
    })

def generate_detailed_report(container):
    """Generate a comprehensive packing report"""
    report = {
        "summary": {
            "container_dimensions": container.dimensions,
            "volume_utilization": f"{container.volume_utilization:.1f}%",
            "total_items_packed": len(container.items),
            "total_weight": f"{container.total_weight:.2f} kg",
            "remaining_volume": f"{container.remaining_volume:.2f} m³",
            "center_of_gravity": [f"{x:.2f}" for x in container.center_of_gravity],
            "weight_balance_score": f"{container._calculate_weight_balance_score():.2f}",
            "interlocking_score": f"{container._calculate_interlocking_score():.2f}"
        },
        "packed_items": [
            {
                "name": item.name,
                "position": [f"{x:.3f}" for x in item.position],
                "dimensions": [f"{x:.3f}" for x in item.dimensions],
                "weight": item.weight,
                "fragility": item.fragility,
                "load_bearing": item.load_bear
            }
            for item in container.items
        ],
        "unpacked_items": [
            {
                "name": name,
                "reason": reason,
                "dimensions": [f"{x:.3f}" for x in item.dimensions],
                "weight": item.weight
            }
            for name, (reason, item) in container.unpacked_reasons.items()
        ],
        "placement_analysis": {
            "layer_distribution": analyze_layer_distribution(container),
            "weight_distribution": container.weight_distribution,
            "stability_analysis": analyze_stability(container)
        }
    }
    return report

def analyze_layer_distribution(container):
    """Analyze how items are distributed in layers"""
    layers = {}
    for item in container.items:
        layer_height = round(item.position[2], 2)
        if layer_height not in layers:
            layers[layer_height] = []
        layers[layer_height].append(item)
    return layers

def analyze_stability(container):
    """Analyze stability of packed items"""
    stability_report = {
        "item_stability": {},
        "overall_stability": 0,
        "critical_points": []
    }
    
    for item in container.items:
        support_score = _calculate_support_score(container, item)
        cog_impact = _calculate_cog_impact(container, item)
        interlocking = _calculate_item_interlocking(container, item)
        
        stability_report["item_stability"][item.name] = {
            "support_score": f"{support_score:.2f}",
            "cog_impact": f"{cog_impact:.2f}",
            "interlocking": f"{interlocking:.2f}",
            "overall": f"{(support_score + cog_impact + interlocking) / 3:.2f}"
        }
        
        if support_score < 0.5 or cog_impact < 0.5:
            stability_report["critical_points"].append({
                "item": item.name,
                "position": [f"{x:.2f}" for x in item.position],
                "issue": "Low stability score"
            })
    
    stability_report["overall_stability"] = f"{sum(float(x['overall']) for x in stability_report['item_stability'].values()) / len(container.items):.2f}"
    return stability_report

def _calculate_support_score(container, item):
    """Calculate how well an item is supported"""
    x, y, z = item.position
    w, d, h = item.dimensions
    
    if z == 0:  # On the ground
        return 1.0
        
    support_area = 0
    total_area = w * d
    
    for other in container.items:
        if other == item:
            continue
            
        if abs(other.position[2] + other.dimensions[2] - z) < 0.001:
            overlap = _calculate_overlap_area(
                (x, y, w, d),
                (other.position[0], other.position[1], 
                 other.dimensions[0], other.dimensions[1])
            )
            support_area += overlap
            
    return min(support_area / total_area, 1.0)

def _calculate_cog_impact(container, item):
    """Calculate impact on center of gravity"""
    ideal_cog = np.array(container.dimensions) / 2
    current_cog = np.array(container.center_of_gravity)
    item_cog = np.array(item.position) + np.array(item.dimensions) / 2
    
    current_dist = np.linalg.norm(current_cog - ideal_cog)
    item_dist = np.linalg.norm(item_cog - ideal_cog)
    
    return 1.0 / (1.0 + abs(item_dist - current_dist))

def _calculate_item_interlocking(container, item):
    """Calculate how well item interlocks with others"""
    contact_count = 0
    max_contacts = 6  # Maximum possible contacts (6 faces)
    
    for other in container.items:
        if other == item:
            continue
            
        if container._has_surface_contact(item.position, item.dimensions, other):
            contact_count += 1
            
    return contact_count / max_contacts

@app.route('/view_report')
def view_report():
    """View detailed packing report"""
    if container_storage.current_container is None:
        return jsonify({'error': 'No container data available'})
        
    container = container_storage.current_container
    report = generate_detailed_report(container)
    
    return render_template('report.html', 
                         report=report,
                         container=container)

@app.route('/preview_csv', methods=['POST'])
def preview_csv():
    """Preview uploaded CSV file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
        
    try:
        df = pd.read_csv(file)
        preview = df.head().to_dict('records')
        columns = df.columns.tolist()
        return jsonify({
            'success': True,
            'preview': preview,
            'columns': columns
        })
    except Exception as e:
        return jsonify({'error': f'Error reading CSV: {str(e)}'})

def _calculate_overlap_area(rect1, rect2):
    """Calculate overlap area between two rectangles"""
    x1, y1, w1, d1 = rect1
    x2, y2, w2, d2 = rect2  # Fixed parameter names to match usage
    
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + d1, y2 + d2) - max(y1, y2))
    
    return x_overlap * y_overlap

def add_unpacked_table(fig, container):
    """Add table showing unpacked items and reasons"""
    if container.unpacked_reasons:
        df = pd.DataFrame([
            {
                'Item': name,
                'Reason': reason,
                'Dimensions': f"{item.dimensions[0]}x{item.dimensions[1]}x{item.dimensions[2]}",
                'Weight': item.weight
            }
            for name, (reason, item) in container.unpacked_reasons.items()
        ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Item', 'Reason', 'Dimensions', 'Weight'],
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=[
                        df['Item'],
                        df['Reason'],
                        df['Dimensions'],
                        df['Weight']
                    ],
                    fill_color='lavender',
                    align='left',
                    font=dict(size=11)
                ),
                columnwidth=[2, 4, 2, 1]
            ),
            row=1, col=2
        )
        
        # Add header for unpacked items section
        fig.update_layout(
            annotations=[
                dict(
                    text="Unpacked Items",
                    xref="paper",
                    yref="paper",
                    x=1.0,
                    y=1.0,
                    xanchor="right",
                    yanchor="bottom",
                    font=dict(size=14),
                    showarrow=False
                )
            ]
        )

# Add monitoring and logging configuration
if not app.debug:
    import logging
    from logging.handlers import RotatingFileHandler
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.mkdir('logs')
        
    # Set up file handler
    file_handler = RotatingFileHandler(
        'logs/container_packing.log', 
        maxBytes=1024 * 1024,  # 1MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    
    app.logger.setLevel(logging.INFO)
    app.logger.info('Container Packing Web App startup')

# Error handling for common exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f'Unhandled exception: {str(e)}')
    return jsonify({
        'error': 'An unexpected error occurred',
        'details': str(e)
    }), 500

# Add API endpoint for getting container statistics
@app.route('/api/container/stats')
def get_container_stats():
    if container_storage.current_container is None:
        return jsonify({'error': 'No container data available'}), 404
        
    container = container_storage.current_container
    return jsonify({
        'dimensions': container.dimensions,
        'volume_utilization': container.volume_utilization,
        'items_packed': len(container.items),
        'total_weight': container.total_weight,
        'center_of_gravity': container.center_of_gravity,
        'weight_balance_score': container._calculate_weight_balance_score(),
        'interlocking_score': container._calculate_interlocking_score()
    })

# Add endpoint for getting item details
@app.route('/api/items/<item_name>')
def get_item_details(item_name):
    if container_storage.current_container is None:
        return jsonify({'error': 'No container data available'}), 404
        
    container = container_storage.current_container
    
    # Search for item in packed items
    for item in container.items:
        if item.name == item_name:
            return jsonify({
                'name': item.name,
                'position': item.position,
                'dimensions': item.dimensions,
                'weight': item.weight,
                'fragility': item.fragility,
                'stackable': item.stackable,
                'boxing_type': item.boxing_type,
                'bundle': item.bundle
            })
            
    # Search in unpacked items
    if item_name in container.unpacked_reasons:
        reason, item = container.unpacked_reasons[item_name]
        return jsonify({
            'name': item.name,
            'dimensions': item.dimensions,
            'weight': item.weight,
            'fragility': item.fragility,
            'stackable': item.stackable,
            'boxing_type': item.boxing_type,
            'bundle': item.bundle,
            'unpacked_reason': reason
        })
        
    return jsonify({'error': 'Item not found'}), 404

# ...existing code...

# Add a new route for checking container status
@app.route('/status')
def get_container_status():
    if container_storage.current_container is None:
        return jsonify({
            'status': 'no_container',
            'message': 'No container has been optimized yet'
        })
    
    container = container_storage.current_container
    return jsonify({
        'status': 'ready',
        'utilization': container.volume_utilization,
        'items_packed': len(container.items),
        'unpacked_items': len(container.unpacked_reasons)
    })

# Add a route for clearing current container
@app.route('/clear', methods=['POST'])
def clear_container():
    container_storage.current_container = None
    container_storage.current_report = None
    return jsonify({'status': 'cleared'})

# Add websocket support for real-time updates
from flask_socketio import SocketIO, emit
socketio = SocketIO(app)

@socketio.on('request_update')
def handle_update_request():
    if container_storage.current_container:
        emit('container_update', {
            'utilization': container_storage.current_container.volume_utilization,
            'items_packed': len(container_storage.current_container.items),
            'visualization': create_interactive_visualization(container_storage.current_container).to_json()
        })

@app.route('/generate_alternative_plan')
def generate_alternative_plan():
    if container_storage.current_container is None:
        return jsonify({'error': 'No container data available'})
    
    try:
        # Generate multiple arrangements
        arrangements = container_storage.current_container.generate_multiple_arrangements(5)
        
        if not arrangements:
            return jsonify({'error': 'Could not generate alternative arrangements'})
        
        # Format results
        alternatives = []
        for container, score in arrangements:
            fig = create_interactive_visualization(container)
            alternatives.append({
                'plot': fig.to_html(),
                'stats': {
                    'utilization': f"{container.volume_utilization:.1f}",
                    'items_packed': len(container.items),
                    'remaining_volume': f"{container.remaining_volume:.2f}",
                    'interlocking_score': f"{container._calculate_interlocking_score():.2f}",
                    'space_efficiency': f"{(1 - (container.remaining_volume / container.total_volume)) * 100:.1f}",
                    'quality_score': f"{score:.2f}"
                }
            })
        
        return jsonify({
            'success': True,
            'alternatives': alternatives
        })
        
    except Exception as e:
        app.logger.error(f"Error generating alternative plans: {str(e)}")
        return jsonify({
            'error': 'Failed to generate alternative arrangements',
            'details': str(e)
        }), 500

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

def create_interactive_visualization(container, container_info=None):
    """Create an interactive 3D visualization of packed items in the container"""
    fig = go.Figure()

    x, y, z = container.dimensions
    
    # Create title with container information
    title_text = 'Container Loading Visualization<br>'
    if container_info:
        title_text += f'Type: {container_info["type"]}<br>'
        title_text += f'Transport Mode: {container_info["transport_mode"]}<br>'
    title_text += f'Dimensions: {x:.2f}m × {y:.2f}m × {z:.2f}m'

    # Add container walls with transparency
    fig.add_trace(go.Mesh3d(
        # 8 vertices of a cube
        x=[0, x, x, 0, 0, x, x, 0],
        y=[0, 0, y, y, 0, 0, y, y],
        z=[0, 0, 0, 0, z, z, z, z],
        i=[0, 0, 0, 1, 4, 4, 4, 5],  # Index of vertices for triangles
        j=[1, 2, 5, 6, 6, 7, 7, 6],
        k=[2, 3, 7, 3, 6, 7, 6, 7],
        opacity=0.2,
        color='lightgrey',
        flatshading=True,
        lighting=dict(
            ambient=0.8,
            diffuse=0.9,
            fresnel=0.2,
            specular=0.5,
            roughness=0.5
        ),
        showlegend=False,
        hoverinfo='none'
    ))

    # Add items with proper 3D box rendering
    for item in container.items:
        x0, y0, z0 = item.position
        dx, dy, dz = item.dimensions

        # Define all 8 vertices of the box
        vertices = [
            [x0, y0, z0], [x0+dx, y0, z0], [x0+dx, y0+dy, z0], [x0, y0+dy, z0],  # bottom
            [x0, y0, z0+dz], [x0+dx, y0, z0+dz], [x0+dx, y0+dy, z0+dz], [x0, y0+dy, z0+dz]  # top
        ]

        # Color based on fragility with better visibility
        if item.fragility == 'HIGH':
            color = 'rgba(255, 99, 71, 0.9)'  # Tomato red
        elif item.fragility == 'MEDIUM':
            color = 'rgba(30, 144, 255, 0.9)'  # Dodger blue
        else:
            color = 'rgba(60, 179, 113, 0.9)'  # Medium sea green

        # Create triangular faces for complete box
        # Each face is made up of multiple triangles
        i = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]  # First vertex index
        j = [1, 2, 5, 6, 6, 7, 7, 4, 5, 6, 6, 7]  # Second vertex index
        k = [2, 3, 6, 7, 7, 4, 4, 5, 1, 2, 2, 3]  # Third vertex index

        # Add box as a mesh with all faces colored
        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in vertices],
            y=[v[1] for v in vertices],
            z=[v[2] for v in vertices],
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=0.95,
            flatshading=True,
            lighting=dict(
                ambient=0.8,
                diffuse=0.9,
                fresnel=0.2,
                specular=0.5,
                roughness=0.5
            ),
            name=item.name,
            showlegend=True,
            hoverinfo='text',
            hovertext=f'{item.name}<br>' +
                     f'Position: ({x0:.2f}, {y0:.2f}, {z0:.2f})<br>' +
                     f'Dimensions: {dx:.2f}×{dy:.2f}×{dz:.2f}<br>' +
                     f'Weight: {item.weight:.2f}kg<br>' +
                     f'Fragility: {item.fragility}'
        ))

        # Add edges for better definition
        edges = [
            # Bottom face
            ([x0, x0+dx], [y0, y0], [z0, z0]),
            ([x0+dx, x0+dx], [y0, y0+dy], [z0, z0]),
            ([x0+dx, x0], [y0+dy, y0+dy], [z0, z0]),
            ([x0, x0], [y0+dy, y0], [z0, z0]),
            # Top face
            ([x0, x0+dx], [y0, y0], [z0+dz, z0+dz]),
            ([x0+dx, x0+dx], [y0, y0+dy], [z0+dz, z0+dz]),
            ([x0+dx, x0], [y0+dy, y0+dy], [z0+dz, z0+dz]),
            ([x0, x0], [y0+dy, y0], [z0+dz, z0+dz]),
            # Vertical edges
            ([x0, x0], [y0, y0], [z0, z0+dz]),
            ([x0+dx, x0+dx], [y0, y0], [z0, z0+dz]),
            ([x0+dx, x0+dx], [y0+dy, y0+dy], [z0, z0+dz]),
            ([x0, x0], [y0+dy, y0+dy], [z0, z0+dz])
        ]

        # Add black edges for better definition
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='none'
            ))

    # Update layout with improved title and annotations
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2, y=2, z=1.5)
            ),
            xaxis=dict(title=f'Length: {x:.2f}m'),
            yaxis=dict(title=f'Width: {y:.2f}m'),
            zaxis=dict(title=f'Height: {z:.2f}m'),
            dragmode='turntable'
        ),
        showlegend=True,
        title=dict(
            text=title_text,
            x=0.5,
            y=0.95
        ),
        margin=dict(l=0, r=0, t=100, b=0)  # Increased top margin for title
    )

    return fig

if __name__ == '__main__':
    try:
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Run with socketio instead of regular Flask run
        socketio.run(app, debug=True)
    except Exception as e:
        app.logger.error(f"Failed to start application: {e}")
        raise
