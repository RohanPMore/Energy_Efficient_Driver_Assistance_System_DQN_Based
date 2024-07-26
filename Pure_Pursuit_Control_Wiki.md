# PurePursuitController Wiki

## Overview

The `PurePursuitController` class implements the Pure Pursuit algorithm for autonomous vehicle path tracking. This controller reads a predefined path from a CSV file and adjusts the vehicle's steering to follow the path. The code also includes a simulation setup using BeamNGpy to test the controller in the BeamNG.drive simulator.

## Class: PurePursuitController

### Initialization

```python
def __init__(self, path_file, lookahead_distance):
```

- **Parameters:**
  - `path_file` (str): The path to the CSV file containing the path data.
  - `lookahead_distance` (float): The distance ahead of the vehicle to consider for the Pure Pursuit algorithm.

- **Attributes:**
  - `self.path` (DataFrame): The path data read from the CSV file.
  - `self.lookahead_distance` (float): The lookahead distance for the Pure Pursuit algorithm.
  - `self.controller_path` (list): List to store the path followed by the controller.

### Methods

#### find_closest_point

```python
def find_closest_point(self, current_position):
```

- **Parameters:**
  - `current_position` (array-like): The current position of the vehicle as `[x, y]`.

- **Returns:**
  - `closest_point_index` (int): The index of the closest point on the path to the current position.

#### get_lookahead_point

```python
def get_lookahead_point(self, current_position, closest_point_index):
```

- **Parameters:**
  - `current_position` (array-like): The current position of the vehicle as `[x, y]`.
  - `closest_point_index` (int): The index of the closest point on the path.

- **Returns:**
  - `lookahead_point` (numpy array): The lookahead point on the path as `[x, y]`.

#### calculate_curvature

```python
def calculate_curvature(self, current_position, lookahead_point):
```

- **Parameters:**
  - `current_position` (array-like): The current position of the vehicle as `[x, y]`.
  - `lookahead_point` (numpy array): The lookahead point on the path as `[x, y]`.

- **Returns:**
  - `curvature` (float): The curvature required to reach the lookahead point.

#### get_lookahead_point_with_curvature

```python
def get_lookahead_point_with_curvature(self, current_position, closest_point_index):
```

- **Parameters:**
  - `current_position` (array-like): The current position of the vehicle as `[x, y]`.
  - `closest_point_index` (int): The index of the closest point on the path.

- **Returns:**
  - `lookahead_point` (numpy array): The lookahead point on the path considering curvature as `[x, y]`.

#### pure_pursuit

```python
def pure_pursuit(self, current_position):
```

- **Parameters:**
  - `current_position` (array-like): The current position of the vehicle as `[x, y]`.

- **Returns:**
  - `delta` (float): The steering angle computed by the Pure Pursuit algorithm.

## Main Execution

### Setup

- Load the path data from the CSV file and initialize the `PurePursuitController`.
- Connect to the BeamNG server and setup the scenario with the vehicle.

### Simulation Loop

- Poll the vehicle's sensors to get its current position.
- Compute the steering angle using the Pure Pursuit algorithm.
- Control the vehicle with the computed steering angle.
- Plot the vehicle's position in real-time to visualize the path following.

### Plotting

- Plot the original path and the vehicle's path in real-time.
- Display the final path followed by the vehicle.

## Example Usage

```python
if __name__ == "__main__":
    path_file = "AMORiPathAndDistanceSpeedLimits.csv"
    lookahead_distance = 0.01
    controller = PurePursuitController(path_file, lookahead_distance)

    bng = BeamNGpy('127.0.0.1', 64535, remote=True)
    bng.open(launch=False)
    scenario = Scenario('smallgrid', 'Test')
    vehicle = Vehicle('ego_vehicle', model='renault_zoe_q90', license='PYTHON')
    scenario.add_vehicle(vehicle, pos=(0, 0, 0))

    scenario.make(bng)
    bng.load_scenario(scenario)
    bng.start_scenario()

    vehicle.set_shift_mode('arcade')
    vehicle.poll_sensors()

    plt.figure(figsize=(8, 6))
    plt.ion()
    plt.plot(controller.path['x'], controller.path['y'], label='Path', color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Final Path Followed by the Controller with Vehicle Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    distance_covered = 0
    previous_position = None

    for i in range(len(controller.path)):
        vehicle_position = np.array([vehicle.state['pos'][0], vehicle.state['pos'][1]])
        if abs(controller.path.loc[i, 'x'] - vehicle_position[0]) > 500 or abs(controller.path.loc[i, 'y'] - vehicle_position[1]) > 500:
            vehicle_position = np.array([controller.path.loc[i, 'x'], controller.path.loc[i, 'y']])

        if previous_position is not None:
            distance_covered += np.sqrt((vehicle_position[0] - previous_position[0])**2 + (vehicle_position[1] - previous_position[1])**2)
        
        previous_position = vehicle_position

        steering_angle = controller.pure_pursuit(vehicle_position)
        vehicle.control(throttle=1, steering=steering_angle)

        print(f"Total distance covered by the vehicle: {distance_covered:.2f} units")
        plt.scatter(vehicle_position[0], vehicle_position[1], label='Vehicle Position', color='red')
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    print(f"Total distance covered by the vehicle: {distance_covered:.2f} units")
```

## Dependencies

- pandas
- numpy
- matplotlib
- math
- time
- beamngpy

## Notes

- Adjust the `lookahead_distance` as needed for your specific use case.
- Ensure the path file is formatted correctly with 'x' and 'y' columns for path coordinates.
- The vehicle model and license plate can be modified as required.