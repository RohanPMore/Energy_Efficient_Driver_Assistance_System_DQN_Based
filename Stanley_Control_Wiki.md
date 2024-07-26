# Stanley Controller for BeamNG Vehicle Simulation

## Overview
This script implements a Stanley controller for autonomous vehicle path following using the BeamNG simulation environment. The Stanley controller is a common control algorithm used for path tracking in autonomous driving. This script reads a predefined path from a CSV file, computes the required steering angles, and controls a vehicle in the BeamNG simulator to follow the path.

## Dependencies
- `pandas`: For handling CSV file operations.
- `numpy`: For numerical operations.
- `matplotlib`: For plotting the path and vehicle positions.
- `beamngpy`: For interfacing with the BeamNG simulation environment.
- `math`: For mathematical operations.
- `time`: For time-related operations.

Ensure you have the necessary libraries installed:
```bash
pip install pandas numpy matplotlib beamngpy
```

## Code Components

### Imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from beamngpy import BeamNGpy, Vehicle, Scenario, Road
```
These are the required imports for the script.

### StanleyController Class
#### Initialization
```python
class StanleyController:
    def __init__(self, path_file, k=1.0, k_soft=0.1):
        self.path = pd.read_csv(path_file)  # Load path data from CSV
        self.k = k  # Gain for the cross-track error
        self.k_soft = k_soft  # Softening term to avoid division by zero
        self.controller_path = []  # Store the path followed by the controller
```
- **Parameters**:
  - `path_file`: Path to the CSV file containing the path data.
  - `k`: Gain for the cross-track error.
  - `k_soft`: Softening term to avoid division by zero.

#### Finding Closest Point on Path
```python
def find_closest_point(self, current_position):
    distances = np.sqrt((self.path['x'] - current_position[0])**2 + (self.path['y'] - current_position[1])**2)
    closest_point_index = np.argmin(distances)
    return closest_point_index
```
- **Parameters**: 
  - `current_position`: Current position of the vehicle.
- **Returns**: Index of the closest point on the path to the vehicle.

#### Computing Heading Error
```python
def compute_heading_error(self, current_position, closest_point_index):
    path_heading = np.arctan2(
        self.path.loc[closest_point_index + 1, 'y'] - self.path.loc[closest_point_index, 'y'],
        self.path.loc[closest_point_index + 1, 'x'] - self.path.loc[closest_point_index, 'x']
    )
    vehicle_heading = np.arctan2(
        current_position[1] - self.path.loc[closest_point_index, 'y'],
        current_position[0] - self.path.loc[closest_point_index, 'x']
    )
    heading_error = path_heading - vehicle_heading
    return heading_error
```
- **Parameters**:
  - `current_position`: Current position of the vehicle.
  - `closest_point_index`: Index of the closest point on the path.
- **Returns**: Heading error between the vehicle's heading and the path's heading.

#### Computing Cross-Track Error
```python
def compute_cross_track_error(self, current_position, closest_point_index):
    closest_point = np.array([self.path.loc[closest_point_index, 'x'], self.path.loc[closest_point_index, 'y']])
    next_point = np.array([self.path.loc[closest_point_index + 1, 'x'], self.path.loc[closest_point_index + 1, 'y']])
    path_vector = next_point - closest_point
    vehicle_vector = current_position - closest_point
    cross_track_error = np.cross(path_vector, vehicle_vector) / np.linalg.norm(path_vector)
    return cross_track_error
```
- **Parameters**:
  - `current_position`: Current position of the vehicle.
  - `closest_point_index`: Index of the closest point on the path.
- **Returns**: Cross-track error (distance of the vehicle from the path).

#### Stanley Control Method
```python
def stanley_control(self, current_position):
    closest_point_index = self.find_closest_point(current_position)
    cross_track_error = self.compute_cross_track_error(current_position, closest_point_index)
    heading_error = self.compute_heading_error(current_position, closest_point_index)

    # Compute steering angle
    steering_angle = heading_error + np.arctan2(self.k * cross_track_error, self.k_soft + np.linalg.norm(current_position))

    # Store the path followed by the controller
    self.controller_path.append(current_position)
    return steering_angle
```
- **Parameters**:
  - `current_position`: Current position of the vehicle.
- **Returns**: Steering angle to correct the vehicle's trajectory based on the Stanley control method.

### Main Execution
```python
if __name__ == "__main__":
    path_file = "AMORiPathAndDistanceSpeedLimits.csv"
    controller = StanleyController(path_file)

    # Connect to the BeamNG server
    bng = BeamNGpy('127.0.0.1', 64535, remote=True)
    bng.open(launch=False)
    scenario = Scenario('smallgrid', 'Test')
    vehicle = Vehicle('ego_vehicle', model='etk800', license='PYTHON')
    scenario.add_vehicle(vehicle, pos=(0, 0, 0))

    # Start the scenario
    scenario.make(bng)
    bng.load_scenario(scenario)
    bng.start_scenario()

    vehicle.set_shift_mode('arcade')
    vehicle.poll_sensors()

    # Initialize plot
    plt.figure(figsize=(8, 6))
    plt.ion()

    # Plot the original path
    plt.plot(controller.path['x'], controller.path['y'], label='Path', color='blue')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Final Path Followed by the Controller with Vehicle Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Simulation loop
    distance_covered = 0
    previous_position = None

    for i in range(len(controller.path)):
        # Extract vehicle position from state
        vehicle.poll_sensors()
        vehicle_position = np.array([vehicle.state['pos'][0], vehicle.state['pos'][1]])

        # Check for sudden changes in coordinates
        if abs(controller.path.loc[i, 'x'] - vehicle_position[0]) > 500 or abs(controller.path.loc[i, 'y'] - vehicle_position[1]) > 500:
            vehicle_position = np.array([controller.path.loc[i, 'x'], controller.path.loc[i, 'y']])

        if previous_position is not None:
            distance_covered += np.sqrt((vehicle_position[0] - previous_position[0])**2 + (vehicle_position[1] - previous_position[1])**2)
        
        previous_position = vehicle_position

        steering_angle = controller.stanley_control(vehicle_position)

        # Update vehicle position and steering angle
        vehicle.control(throttle=0.5, steering=steering_angle)

        # Plot vehicle positions
        plt.scatter(vehicle_position[0], vehicle_position[1], label='Vehicle Position', color='red')
        plt.pause(0.01)

        print(f"Current position: {vehicle_position}, Steering angle: {steering_angle:.2f}")
    
    plt.ioff()
    plt.show()

    print(f"Total distance covered by the vehicle: {distance_covered:.2f} units")

    bng.close()
```
### Explanation:
- The script initializes a `StanleyController` with a path file.
- It connects to the BeamNG simulator and sets up a scenario with a vehicle.
- The script then enters a simulation loop where it continuously updates the vehicle's position and computes the required steering angle using the Stanley control method.
- The vehicle's path and position are plotted in real-time.
- The total distance covered by the vehicle is printed at the end of the simulation.

### Notes:
- Ensure the BeamNG simulator is running and accessible at the specified IP address and port.
- Adjust the `path_file` variable to point to your specific CSV file containing the path data.
- Modify the `throttle` value in the `vehicle.control` method as needed to control the vehicle's speed.