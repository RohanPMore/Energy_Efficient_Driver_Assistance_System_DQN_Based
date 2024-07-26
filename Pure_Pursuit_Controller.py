import pandas as pd  # Library for data manipulation and analysis, used here for reading CSV files
import numpy as np  # Library for numerical computations, used here for calculations
import matplotlib.pyplot as plt  # Library for plotting graphs
import math  # Standard math library, used here for mathematical operations
import time  # Standard time library, used here for time-related operations
from beamngpy import BeamNGpy, Vehicle, Scenario, Road  # BeamNGpy library for interacting with the BeamNG simulator

# Class to implement the Pure Pursuit controller
class PurePursuitController:
    def __init__(self, path_file, lookahead_distance):
        self.path = pd.read_csv(path_file)  # Read the entire path from the CSV file
        self.lookahead_distance = lookahead_distance  # Distance to look ahead on the path
        self.controller_path = []  # Initialize an empty list to store the path followed by the controller

    # Find the closest point on the path to the current position
    def find_closest_point(self, current_position):
        distances = np.sqrt((self.path['x'] - current_position[0])**2 + (self.path['y'] - current_position[1])**2)  # Calculate distances to all path points
        closest_point_index = np.argmin(distances)  # Find the index of the closest point
        return closest_point_index

    # Get the lookahead point on the path
    def get_lookahead_point(self, current_position, closest_point_index):
        distances = np.sqrt((self.path['x'] - current_position[0])**2 + (self.path['y'] - current_position[1])**2)  # Calculate distances to all path points
        remaining_distances = distances[closest_point_index:]  # Distances from the closest point onward
        total_distance = np.cumsum(remaining_distances)  # Cumulative distance from the closest point
        lookahead_index = np.argmax(total_distance > self.lookahead_distance) + closest_point_index  # Find the index of the lookahead point
        lookahead_point = np.array([self.path.loc[lookahead_index, 'x'], self.path.loc[lookahead_index, 'y']])  # Coordinates of the lookahead point
        return lookahead_point

    # Calculate the curvature required for the pure pursuit controller
    def calculate_curvature(self, current_position, lookahead_point):
        x, y = current_position
        x1, y1 = lookahead_point
        distance = np.sqrt((x1 - x)**2 + (y1 - y)**2)  # Distance to the lookahead point
        if distance == 0:
            return 0  # Avoid division by zero
        curvature = 2 * (x1 - x) / distance**2  # Calculate curvature
        return curvature

    # Get the lookahead point considering curvature
    def get_lookahead_point_with_curvature(self, current_position, closest_point_index):
        lookahead_point = self.get_lookahead_point(current_position, closest_point_index)  # Get the initial lookahead point
        curvature = self.calculate_curvature(current_position, lookahead_point)  # Calculate the curvature
        
        # Adjust lookahead distance based on curvature
        if curvature != 0:
            lookahead_distance_adjusted = min(self.lookahead_distance, 1 / abs(curvature))  # Adjust the lookahead distance
        else:
            lookahead_distance_adjusted = self.lookahead_distance
        
        total_distance = 0
        lookahead_index = closest_point_index
        for i in range(closest_point_index, len(self.path)):
            total_distance += np.sqrt((self.path.loc[i, 'x'] - current_position[0])**2 + (self.path.loc[i, 'y'] - current_position[1])**2)  # Accumulate distances
            if total_distance >= lookahead_distance_adjusted:
                lookahead_index = i
                break
        lookahead_point = np.array([self.path.loc[lookahead_index, 'x'], self.path.loc[lookahead_index, 'y']])  # Adjusted lookahead point
        return lookahead_point

    # Pure pursuit control method to compute the steering angle
    def pure_pursuit(self, current_position):
        closest_point_index = self.find_closest_point(current_position)  # Find the closest point on the path
        lookahead_point = self.get_lookahead_point_with_curvature(current_position, closest_point_index)  # Get the lookahead point with curvature
        
        # Calculate the steering angle using the pure pursuit algorithm
        L = np.sqrt((lookahead_point[0] - current_position[0])**2 + (lookahead_point[1] - current_position[1])**2)  # Distance to the lookahead point
        x, y = current_position
        x1, y1 = lookahead_point
        alpha = np.arctan2(y1 - y, x1 - x)  # Angle to the lookahead point
        delta = np.arctan2(2 * 1.5 * np.sin(alpha), L)  # Steering angle calculation
        
        # Store the path followed by the controller
        self.controller_path.append(current_position)

        return delta

# Main execution block
if __name__ == "__main__":
    path_file = "AMORiPathAndDistanceSpeedLimits.csv"  # Path to the CSV file containing the path data
    lookahead_distance = 0.01  # Lookahead distance for the pure pursuit controller
    controller = PurePursuitController(path_file, lookahead_distance)  # Initialize the controller

    # Connect to the BeamNG server
    bng = BeamNGpy('127.0.0.1', 64535, remote=True)  # Connect to BeamNG simulator
    bng.open(launch=False)
    scenario = Scenario('smallgrid', 'Test')  # Create a new scenario in the simulator
    vehicle = Vehicle('ego_vehicle', model='renault_zoe_q90', license='PYTHON')  # Define the vehicle
    scenario.add_vehicle(vehicle, pos=(0, 0, 0))  # Add the vehicle to the scenario

    # Start the scenario
    scenario.make(bng)  # Make the scenario
    bng.load_scenario(scenario)  # Load the scenario in the simulator
    bng.start_scenario()  # Start the scenario

    vehicle.set_shift_mode('arcade')  # Set vehicle mode to 'arcade'
    vehicle.poll_sensors()  # Poll sensors to get the initial state

    # Initialize plot
    plt.figure(figsize=(8, 6))  # Create a new figure for plotting
    plt.ion()  # Turn on interactive mode

    # Plot the original path
    plt.plot(controller.path['x'], controller.path['y'], label='Path', color='blue')  # Plot the path from the CSV file

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Final Path Followed by the Controller with Vehicle Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Simulation loop
    distance_covered = 0  # Initialize distance covered
    previous_position = None  # Initialize previous position

    for i in range(len(controller.path)):  # Loop over the path points
        # Extract vehicle position from state
        vehicle_position = np.array([vehicle.state['pos'][0], vehicle.state['pos'][1]])  # Get the current vehicle position

        # Check for sudden changes in coordinates
        if abs(controller.path.loc[i, 'x'] - vehicle_position[0]) > 500 or abs(controller.path.loc[i, 'y'] - vehicle_position[1]) > 500:
            vehicle_position = np.array([controller.path.loc[i, 'x'], controller.path.loc[i, 'y']])  # Correct sudden changes

        if previous_position is not None:
            distance_covered += np.sqrt((vehicle_position[0] - previous_position[0])**2 + (vehicle_position[1] - previous_position[1])**2)  # Calculate distance covered
        
        previous_position = vehicle_position  # Update previous position

        steering_angle = controller.pure_pursuit(vehicle_position)  # Get the steering angle from the pure pursuit controller

        # Update vehicle position and steering angle
        vehicle.control(throttle=1, steering=steering_angle)  # Control the vehicle with the computed steering angle

        # Plot vehicle positions
        print(f"Total distance covered by the vehicle: {distance_covered:.2f} units")  # Print distance covered
        plt.scatter(vehicle_position[0], vehicle_position[1], label='Vehicle Position', color='red')  # Plot the vehicle position
        plt.pause(0.01)  # Pause to update the plot

    plt.ioff()  # Turn off interactive mode after the loop
    plt.show()  # Display the final plot

    print(f"Total distance covered by the vehicle: {distance_covered:.2f} units")  # Print the total distance covered
