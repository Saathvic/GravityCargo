import sys
import os
import pandas as pd
import pyvista as pv
import numpy as np
import requests
import re

# Fixed Weather API Key
WEATHER_API_KEY = "c28414184f314651876190006242812"

# Route and Weather Functions

def get_mock_routes(source, destination):
    """Generate mock route data"""
    return f"""
    TRUCK-FRIENDLY
    Route 1:
    * Total Distance: 134 km
    * Expected Duration: 2 hours 30 minutes
    * Key waypoints: Sriperumbudur, Walajapet, Vellore
    * Truck accessibility: Yes

    NOT SUITABLE FOR TRUCKS
    Route 2:
    * Total Distance: 151 km
    * Expected Duration: 3 hours
    * Key waypoints: Mamallapuram, Kalpakkam, Chengalpet
    * Truck accessibility: No
    """

def get_weather_for_waypoint(waypoint):
    """Fetch weather data for a single waypoint"""
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={waypoint}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "current" in data:
                return {
                    "waypoint": waypoint,
                    "temperature": data["current"]["temp_c"],
                    "weather": data["current"]["condition"]["text"],
                    "humidity": data["current"]["humidity"],
                    "wind_speed": data["current"]["wind_kph"]
                }
        return {
            "waypoint": waypoint,
            "temperature": None,
            "weather": "Data unavailable",
            "humidity": None,
            "wind_speed": None
        }
    except Exception as e:
        print(f"Error fetching weather for {waypoint}: {str(e)}")
        return None

def get_route_info(source, destination):
    """Extract route information from mock data"""
    try:
        response_text = get_mock_routes(source, destination)
        # Simplified regex pattern
        pattern = r"TRUCK-FRIENDLY[\s\S]*?Total Distance: (\d+) km[\s\S]*?Duration: ([^\n]*?)[\s\S]*?Key waypoints: ([^\n]*?)[\s\S]*?Truck accessibility: Yes"
        truck_routes = re.findall(pattern, response_text)
        
        if not truck_routes:
            print("No valid routes found in the response")
            return None
            
        shortest_route = min(truck_routes, key=lambda x: int(x[0]))
        distance, duration, waypoints = shortest_route
        return {
            "distance": distance,
            "duration": duration.strip(),
            "waypoints": [wp.strip() for wp in waypoints.split(",")]
        }
    except Exception as e:
        print(f"Error in route info extraction: {e}")
        return None

def get_weather_data(waypoints):
    """Fetch weather data for all waypoints"""
    return [get_weather_for_waypoint(wp) for wp in waypoints]

def calculate_average_temperature(weather_data):
    """Calculate average temperature from weather data"""
    valid_temps = [data["temperature"] for data in weather_data if data and data["temperature"] is not None]
    if not valid_temps:
        return None
    return sum(valid_temps) / len(valid_temps)

# Package Sorting Functions

def validate_and_load_data(file_path):
    """Load and validate data from Excel file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    data = pd.read_excel(file_path)
    required_columns = ['Box ID', 'Dimensions (cm)', 'Quantity', 'Temperature Recommended']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {', '.join(required_columns)}")
    data['Dimensions (cm)'] = data['Dimensions (cm)'].apply(lambda dim: list(map(int, dim.split('x'))))
    return data

def get_box_color(temperature):
    """Return color based on temperature recommendation"""
    if "cool" in str(temperature).lower():
        return '#1E90FF'  # Cool storage
    return '#C0C0C0'  # Regular storage

def create_boxes(data, plotter):
    """Create and visualize boxes within the container"""
    container_length, container_width, container_height = 606, 244, 259
    offset_x = offset_y = offset_z = 0
    for _, row in data.iterrows():
        dims = row['Dimensions (cm)']
        length, width, height = dims
        temperature = row['Temperature Recommended']
        for _ in range(row['Quantity']):
            if offset_x + length > container_length:
                offset_x = 0
                offset_y += width
                if offset_y + width > container_width:
                    offset_y = 0
                    offset_z += height
            if offset_z + height > container_height:
                print(f"Box {row['Box ID']} could not be placed due to height limit.")
                continue
            box = pv.Box(bounds=(
                offset_x, offset_x + length,
                offset_y, offset_y + width,
                offset_z, offset_z + height
            ))
            plotter.add_mesh(box, color=get_box_color(temperature), opacity=0.6)
            offset_x += length

def visualize_packages(data):
    """Visualize packages in 3D container"""
    plotter = pv.Plotter()
    plotter.set_background("white")
    create_boxes(data, plotter)
    plotter.show()

# Integrated Functionality

def main():
    try:
        # Check if Excel file exists before starting
        if not os.path.exists("Larger_Box_Dimensions.xlsx"):
            print("Error: Larger_Box_Dimensions.xlsx not found in current directory")
            return
            
        source = input("Enter source location: ").strip()
        destination = input("Enter destination: ").strip()
        
        if not source or not destination:
            print("Error: Source and destination cannot be empty")
            return
        
        # Get route and weather info
        route_info = get_route_info(source, destination)
        if not route_info:
            print("No truck-friendly routes found.")
            return
        
        weather_data = get_weather_data(route_info['waypoints'])
        if not weather_data:
            print("Unable to fetch weather data.")
            return
            
        avg_temp = calculate_average_temperature(weather_data)
        
        print("\nRoute Information:")
        print(f"Distance: {route_info['distance']} km")
        print(f"Duration: {route_info['duration']}")
        print(f"Waypoints: {', '.join(route_info['waypoints'])}")
        print(f"Average Temperature: {avg_temp:.2f}°C" if avg_temp else "Average Temperature: N/A")
        
        # Load and visualize package data
        print("\nLoading and visualizing package data...")
        data = validate_and_load_data("Larger_Box_Dimensions.xlsx")
        visualize_packages(data)
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":  # Fixed main check
    main()
