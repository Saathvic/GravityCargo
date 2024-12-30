import requests
import re
import pandas as pd
import pyvista as pv
import os
import numpy as np
import sys
from checkpoints import generate_formatted_routes  # Updated import
import google.generativeai as genai

# API Keys
WEATHER_API_KEY = "b8648db03f74408b9f7190038242812"  # New API key
GOOGLE_API_KEY = "AIzaSyB0tRWaZXYa-fC-_dAHBBvRTQkiMolpukI"

# Configure Google AI
genai.configure(api_key=GOOGLE_API_KEY)
def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        'pandas': 'pandas',
        'pyvista': 'pyvista',
        'openpyxl': 'openpyxl'
    }

    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)

    if missing_packages:
        print("Missing required packages. Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)

# Check dependencies first
check_dependencies()

# Route and weather functionality
def get_weather_for_waypoint(waypoint, api_key):
    """Fetch weather data with better handling of Indian city names"""
    try:
        # Add delay to prevent rate limiting
        import time
        time.sleep(1)
        
        # Format location name
        location = f"{waypoint}, Tamil Nadu, India"
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
        
        print(f"Requesting weather for {location}...")  # Debug print
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if "current" in data:
                weather_info = {
                    "waypoint": waypoint,
                    "temperature": data["current"]["temp_c"],
                    "weather": data["current"]["condition"]["text"],
                    "humidity": data["current"]["humidity"],
                    "wind_speed": data["current"]["wind_kph"]
                }
                print(f"Success: {waypoint} - {weather_info['temperature']}°C")
                return weather_info
        elif response.status_code == 403:
            print(f"API Key Error: Please check your Weather API key validity")
            return None
        else:
            print(f"Error {response.status_code} for {waypoint}: {response.text}")
        return None
    except Exception as e:
        print(f"Error fetching weather for {waypoint}: {e}")
        return None

def get_ai_routes(source, destination):
    """Generate route information using Google AI"""
    try:
        # Use the imported function directly
        return generate_formatted_routes(source, destination, GOOGLE_API_KEY)
    except Exception as e:
        print(f"AI route generation failed: {e}")
        return None

def get_route_info(source, destination):
    """Extract route information from AI"""
    try:
        # Get routes using the updated get_ai_routes
        response_text = get_ai_routes(source, destination)
        if not response_text:
            print("Could not generate route information.")
            return None
            
        pattern = r"TRUCK-FRIENDLY\s*Route \d+:\s*\* Total Distance: (\d+) km\s*\* Expected Duration: (.*?)\s*\* Key waypoints: ([\w\s,]+)\s*\* Truck accessibility: Yes"
        truck_routes = re.findall(pattern, response_text, re.DOTALL)
        
        if truck_routes:
            distance, duration, waypoints = truck_routes[0]
            return {
                "distance": distance,
                "duration": duration.strip(),
                "waypoints": [wp.strip() for wp in waypoints.split(",")]
            }
        return None
    except Exception as e:
        print(f"Error in route extraction: {e}")
        return None

def get_weather_data(waypoints, weather_api_key):
    """Improved weather data collection with better error handling"""
    weather_data = []
    for waypoint in waypoints:
        print(f"Fetching weather for {waypoint}...")  # Debug print
        data = get_weather_for_waypoint(waypoint, weather_api_key)
        if data:
            weather_data.append(data)
            print(f"Temperature in {waypoint}: {data['temperature']}°C")
        else:
            print(f"Could not fetch weather for {waypoint}")
    return weather_data

def calculate_average_temperature(weather_data):
    """Calculate average temperature with proper error handling"""
    try:
        valid_temps = [data["temperature"] for data in weather_data if data and data.get("temperature") is not None]
        return sum(valid_temps) / len(valid_temps) if valid_temps else 25.0  # Default to 25°C if no data
    except Exception as e:
        print(f"Error calculating average temperature: {e}")
        return 25.0  # Default temperature

# Package visualization functionality
container_length = 606
container_width = 244
container_height = 259

def validate_and_load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    data = pd.read_excel(file_path)
    required_columns = ['Box ID', 'Dimensions (cm)', 'Quantity', 'Temperature Recommended']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    def parse_dimensions(dim_str):
        try:
            return [int(d.strip()) for d in dim_str.split('x')]
        except Exception:
            raise ValueError(f"Invalid dimension format: {dim_str}. Expected format: LxWxH")

    data['Dimensions (cm)'] = data['Dimensions (cm)'].apply(parse_dimensions)
    return data

def get_box_color(temperature, avg_temperature):
    """Get box color with safe temperature handling"""
    try:
        # Default to 25°C if avg_temperature is None
        avg_temp = avg_temperature if avg_temperature is not None else 25.0
        
        if 'cool' in str(temperature).lower() or avg_temp < 20:
            return '#1E90FF'  # Cool products
        return '#C0C0C0'  # Regular products
    except Exception as e:
        print(f"Error determining box color: {e}")
        return '#C0C0C0'  # Default to regular color

def create_boxes(data, plotter, avg_temperature):
    offset_x = offset_y = offset_z = 0

    for _, row in data.iterrows():
        dims = row['Dimensions (cm)']
        length, width, height = dims
        quantity = row['Quantity']
        temperature = row['Temperature Recommended']

        for _ in range(quantity):
            if offset_x + length > container_length:
                offset_x = 0
                offset_y += width
                if offset_y + width > container_width:
                    offset_y = 0
                    offset_z += height

            if offset_z + height > container_height:
                continue

            box = pv.Box(bounds=(
                offset_x, offset_x + length,
                offset_y, offset_y + width,
                offset_z, offset_z + height
            ))
            color = get_box_color(temperature, avg_temperature)
            plotter.add_mesh(box, color=color, opacity=0.6)

            offset_x += length

def main():
    try:
        # Check Excel file first
        if not os.path.exists("Larger_Box_Dimensions.xlsx"):
            print("Error: Larger_Box_Dimensions.xlsx not found in current directory")
            return
            
        source = input("Enter source location: ").strip()
        destination = input("Enter destination: ").strip()
        
        if not source or not destination:
            print("Error: Source and destination cannot be empty")
            return

        # Get route info
        route_info = get_route_info(source, destination)
        if not route_info:
            print("Could not generate route information.")
            return

        print("\nRoute found:")
        print(f"Distance: {route_info['distance']} km")
        print(f"Duration: {route_info['duration']}")
        print(f"Waypoints: {', '.join(route_info['waypoints'])}")

        print("\nFetching weather data...")
        weather_data = get_weather_data(route_info["waypoints"], WEATHER_API_KEY)  # Use constant
        
        if not weather_data:
            print("Warning: Could not fetch weather data, using default temperature")
            
        avg_temperature = calculate_average_temperature(weather_data)
        print(f"Average temperature along route: {avg_temperature:.1f}°C")

        # Load package data and create visualization
        data = validate_and_load_data("Larger_Box_Dimensions.xlsx")
        
        # Initialize visualization
        plotter = pv.Plotter()
        plotter.set_background('white')
        plotter.add_camera_orientation_widget()

        # Add container
        container = pv.Box(bounds=(0, container_length, 0, container_width, 0, container_height))
        plotter.add_mesh(container, color='blue', opacity=0.2, line_width=2)

        # Create boxes with safe temperature handling
        create_boxes(data, plotter, avg_temperature)
        
        # Add temperature information to display
        plotter.add_text(
            f"Route Temperature: {avg_temperature:.1f}°C\nBlue: Cool Products\nGray: Regular Products",
            position='upper_right',
            font_size=10,
            color='black'
        )

        plotter.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":  # Fixed the main check
    main()