import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import math  # For mathematical operations
import time  # For time-related functions
from beamngpy import BeamNGpy, Vehicle, Scenario, Road  # BeamNG library for simulation

# Define a class for the Stanley Controller
class StanleyController:
    def __init__(self, path_file, k=1.0, k_soft=0.1):
        # Initialize the controller with a path file and gain parameters
        self.path = pd.read_csv(path_file)  # Load the path from a CSV file
        self.k = k  # Gain for the cross-track error
        self.k_soft = k_soft  # Softening term to avoid division by zero in arctan
        self.controller_path = []  # Store the path followed by the controller

    def find_closest_point(self, current_position):
        # Find the closest point on the path to the current position of the vehicle
        distances = np.sqrt((self.path['x'] - current_position[0])**2 + (self.path['y'] - current_position[1])**2)
        closest_point_index = np.argmin(distances)  # Index of the closest point
        return closest_point_index

    def compute_heading_error(self, current_position, closest_point_index):
        # Compute the heading error between the vehicle's heading and the path's heading
        path_heading = np.arctan2(
            self.path.loc[closest_point_index + 1, 'y'] - self.path.loc[closest_point_index, 'y'],
            self.path.loc[closest_point_index + 1, 'x'] - self.path.loc[closest_point_index, 'x']
        )
        vehicle_heading = np.arctan2(
            current_position[1] - self.path.loc[closest_point_index, 'y'],
            current_position[0] - self.path.loc[closest_point_index, 'x']
        )
        heading_error = path_heading - vehicle_heading  # Difference between path heading and vehicle heading
        return heading_error

    def compute_cross_track_error(self, current_position, closest_point_index):
        # Compute the cross-track error (distance of the vehicle from the path)
        closest_point = np.array([self.path.loc[closest_point_index, 'x'], self.path.loc[closest_point_index, 'y']])
        next_point = np.array([self.path.loc[closest_point_index + 1, 'x'], self.path.loc[closest_point_index + 1, 'y']])
        path_vector = next_point - closest_point  # Vector along the path segment
        vehicle_vector = current_position - closest_point  # Vector from the closest point to the vehicle
        cross_track_error = np.cross(path_vector, vehicle_vector) / np.linalg.norm(path_vector)  # Perpendicular distance
        return cross_track_error

    def stanley_control(self, current_position):
        # Main Stanley control function to compute the steering angle
        closest_point_index = self.find_closest_point(current_position)  # Find the closest point on the path
        cross_track_error = self.compute_cross_track_error(current_position, closest_point_index)  # Compute cross-track error
        heading_error = self.compute_heading_error(current_position, closest_point_index)  # Compute heading error

        # Compute the steering angle using the Stanley method
        steering_angle = heading_error + np.arctan2(self.k * cross_track_error, self.k_soft + np.linalg.norm(current_position))

        # Store the path followed by the controller
        self.controller_path.append(current_position)
        return steering_angle  # Return the computed steering angle

if __name__ == "__main__":
    path_file = "AMORiPathAndDistanceSpeedLimits.csv"  # Path to the CSV file with the path data
    controller = StanleyController(path_file)  # Initialize the controller

    # Connect to the BeamNG server
    bng = BeamNGpy('127.0.0.1', 64535, remote=True)
    bng.open(launch=False)  # Open the connection to the BeamNG server
    scenario = Scenario('smallgrid', 'Test')  # Create a new scenario
    vehicle = Vehicle('ego_vehicle', model='etk800', license='PYTHON')  # Create a new vehicle
    scenario.add_vehicle(vehicle, pos=(0, 0, 0))  # Add the vehicle to the scenario

    # Start the scenario
    scenario.make(bng)  # Generate the scenario
    bng.load_scenario(scenario)  # Load the scenario
    bng.start_scenario()  # Start the scenario

    vehicle.set_shift_mode('arcade')  # Set the vehicle's transmission mode
    vehicle.poll_sensors()  # Poll the vehicle's sensors

    # Initialize the plot for visualization
    plt.figure(figsize=(8, 6))
    plt.ion()  # Turn on interactive mode for dynamic plotting

    # Plot the original path
    plt.plot(controller.path['x'], controller.path['y'], label='Path', color='blue')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Final Path Followed by the Controller with Vehicle Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Simulation loop
    distance_covered = 0  # Initialize distance covered by the vehicle
    previous_position = None  # Initialize previous position of the vehicle

    for i in range(len(controller.path)):
        # Extract vehicle position from state
        vehicle.poll_sensors()  # Poll the vehicle's sensors
        vehicle_position = np.array([vehicle.state['pos'][0], vehicle.state['pos'][1]])  # Get the current position

        # Check for sudden changes in coordinates to correct the vehicle position
        if abs(controller.path.loc[i, 'x'] - vehicle_position[0]) > 500 or abs(controller.path.loc[i, 'y'] - vehicle_position[1]) > 500:
            vehicle_position = np.array([controller.path.loc[i, 'x'], controller.path.loc[i, 'y']])

        # Compute the distance covered
        if previous_position is not None:
            distance_covered += np.sqrt((vehicle_position[0] - previous_position[0])**2 + (vehicle_position[1] - previous_position[1])**2)
        
        previous_position = vehicle_position  # Update previous position

        # Compute the steering angle using the Stanley controller
        steering_angle = controller.stanley_control(vehicle_position)

        # Update vehicle position and steering angle
        vehicle.control(throttle=0.5, steering=steering_angle)  # Control the vehicle with computed steering angle

        # Plot vehicle positions for visualization
        plt.scatter(vehicle_position[0], vehicle_position[1], label='Vehicle Position', color='red')
        plt.pause(0.01)  # Pause to update the plot

        print(f"Current position: {vehicle_position}, Steering angle: {steering_angle:.2f}")

    plt.ioff()  # Turn off interactive mode after the loop
    plt.show()  # Display the final plot

    print(f"Total distance covered by the vehicle: {distance_covered:.2f} units")  # Print total distance covered

    bng.close()  # Close the connection to the BeamNG server
