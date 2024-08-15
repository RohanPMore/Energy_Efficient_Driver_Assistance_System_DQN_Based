from enum import Enum

# cost_function_enum
class CostFunction(Enum):
    ENERGY_EFFICIENCY = 1
    TIME_EFFICIENCY = 2
    INVERSE_ENERGY_EFFICIENCY = 3
    MINIMAL_ENERGY_EFFICIENCY = 4

class Constants:
    ##-------------------------- Electric Vehicle and Route Parameters -----------------------##
    wheel_radius__m = 0.31075  # m, for Zoe Intens Q90 base variant before 2019
    vehicle_empty_weight__kg = 1318  # kg, weight without driver
    driver_weight__kg = 85  # kg
    vehicle_weight__kg = vehicle_empty_weight__kg + driver_weight__kg  # kg
    rotating_weight__kg = 200  # kg, weight of the rotating mass
    total_weight__kg = vehicle_weight__kg + rotating_weight__kg  # kg
    gear_ratio = 9.3  # transmission ratio of the gear box
    max_braking_torque__Nm = -4360/9.3 # From BeamNG 4360  %-466; 100% of 14kN, multiplied by wheel radius & divided by gear ratio
    gear_box_efficiency = 0.92  # efficiency of the gear box
    motor_max_revolution__rev_per_min = 11190  # max motor revolution speed for Zoe
    motor_min_revolution__rev_per_min = 650  # min revolution speed of the motor for recuperation, rpm
    min_acceleration__m_per_s2 = 0.0001
    min_velocity_m_per_s = 0.05
    
    ##-------------------------------- Battery Cell Parameters -------------------------------##
    reference_current__A = 32.5  # A, for Zoe from LG E63 Datasheet
    reference_capacity__F = 64.8 * 3600  # As, reference battery cell capacity from datasheet
    max_inverter_current__A = 300  # max DC-current for the inverter, A
    cutoff_voltage__V = 2.5  # cutoff voltage, V from LG E63 datasheet
    max_voltage__V = 4.2  # max voltage of the inverter, V
    
    ##----------------------------------- Lookup tables ---------------------------------------##
    state_of_discharge_lookup = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    discharge_resistance_lookup__ohm = [0.00175, 0.00180, 0.00182, 0.00183, 0.00184, 0.00178, 0.00164, 0.00172, # From Discharge resistance table
                                        0.00184, 0.00211, 0.00396, 0.00721, 0.01166]
    charge_resistance_lookup__ohm = [0.00186, 0.00175, 0.00173, 0.00172, 0.00171, 0.00168, 0.00164, 0.00160, # From Discharge resistance table
                                    0.00166, 0.00184, 0.00196, 0.00212, 0.00335]
    # build the lookup table of the max. discharging power of the battery, from discharge power table E63
    max_discharge_power_lookup__W = [275, 490, 535, 550, 555, 560, 565, 570, 580, 590, 600, 620, 650, 660, 680]
    state_of_charge_discharge_lookup = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    # build the lookup table of the max. charging power of the battery from regen power table E63
    max_power_charge_lookup__W = [295, 315, 320, 325, 327, 330, 332, 335, 337, 345, 350, 355, 362, 250, 60]
    state_of_charge_charge_lookup = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    # SOC-OCV Table from LG E63 Battery
    state_of_charge_open_circuit_voltage_lookup = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                                        0.8, 0.85, 0.9, 0.95, 1]
    open_circuit_voltage_lookup__V = [3.167, 3.413, 3.446, 3.488, 3.537, 3.571, 3.593, 3.61, 3.625, 3.642, 3.663,
                                        3.695, 3.755, 3.799, 3.846, 3.895, 3.945, 3.997, 4.051, 4.108, 4.166]
    
    ##---------------------------------- Parameters of the Auxiliary System ----------------------------------##
    touch_screen_radio_state = 1
    touch_screen_radio_power__W = 50  # W
    steering_state = 1
    steering_power__W = 200  # W
    roll_stabilizer_state = 0
    roll_stabilizer_power__W = 150  # W
    air_cooling_state = 0
    air_cooling_power__W = 700  # W
    heating_state = 0
    heating_power__W = 1500  # W
    motor_cooling_state = 1
    motor_cooling_power__W = 100  # W
    low_beam_state = 0
    low_beam_power__W = 90  # W
    
    ##------------------------------------------ Other Parameters ----------------------------------------------##
    air_density__kg_per_m3 = 1.202  # kg/m^3, air density
    front_area__m2 = 2.5862  # m^2, frontal area of the vehicle
    aero_drag_coeff = 0.29  # aerodynamic drag coefficient
    wind_velocity__m_per_s = 0  # m/s, opposing wind speed
    rolling_friction_resistance_coeff = 0.011  # rolling friction resistance coefficient
    gravitational_acceleration__m_per_s2 = 9.81  # m/s^2, gravitational acceleration
    
    ##---------------------------------------- Motor Performance Data ------------------------------------------##
    max_motor_revolution_lookup__rev_per_min = [0, 100, 200, 310, 600, 1260, 2910, 3110, 3300, 3480, 3660, 3830, 3990, 4130, 4290, 4430,
                            4560, 4700, 4820, 5530, 6150, 6690, 7170, 7610, 8000, 8350, 8700, 9020, 9320, 9590,
                            9860, 10090, 10330, 10550, 10780, 10980, 11010, 11070, 11090, 11130, 11150, 11160,
                            11170, 11180, 11190]
    max_negative_motor_torque_lookup__Nm = [0, 0, 0, 0, -118, -118, -118, -118, -118, -116, -109.887, -104.78625, -100.766,
                                -97.2487, -93.622, -90.717, -88.02, -85.445, -83.351, -72.606, -65.2475, -59.9892,
                                -55.9825, -52.7405, -50.165, -48.0587, -46.12, -44.4655, -43.048, -41.8435,
                                -40.6885, -39.7627, -38.839, -38.025, -37.219, -36.529, -36.4357, -36.2905,
                                -36.2421, -36.1452, -36.0968, -36.0726, -36.0484, -36.0242, -36]
    max_motor_torque_lookup___Nm = [225, 225, 225, 225, 225, 222, 225, 218.5, 206.5, 194.5, 182.5, 173.5, 166.5, 159.5,
                            153.5, 148.5, 143.5, 139.5, 135.5, 117.5, 105, 96.5, 89, 83.5, 78.5, 75, 72, 69.5, 67,
                            65.5, 63.5, 62, 60.5, 59, 58, 57, 56, 47, 40, 34.5, 31.5, 29, 27, 12, 0]
    min_negative_motor_revolution_lookup__r_min = [0, 600, 1000, 1400, 1800, 2200, 2600, 3000, 3400, 3800, 4200, 4600, 5000, 5400,
                                5800, 6200, 6600, 7000, 7400, 7800, 8200, 8600, 9000, 9400, 9800, 10200, 10600,
                                11000, 11190]
    min_negative_motor_torque_lookup__Nm = [0, 0, -117.96, -117.96, -117.96, -99.83, -84.47, -73.21, -64.60, -57.80, -52.29,
                                -47.75, -43.93, -40.67, -37.87, -35.42, -33.28, -31.38, -29.68, -28.16, -26.78,
                                -25.54, -24.40, -23.37, -22.41, -21.53, -20.72, -19.97, -19.5]
    
    ##---------------------------------------------- Efficiency values ---------------------------------------------------##
    motor_efficiency_torque_lookup__Nm = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    motor_efficiency_motor_revolution__rev_per_min = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
    motor_efficiency_lookup = [
        [0.5, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
        [0.5, 0.89, 0.91, 0.90, 0.89, 0.86, 0.84, 0.83, 0.82, 0.79, 0.75],
        [0.5, 0.90, 0.94, 0.93, 0.92, 0.91, 0.90, 0.90, 0.89, 0.88, 0.87],
        [0.5, 0.90, 0.95, 0.94, 0.93, 0.92, 0.92, 0.91, 0.90, 0.90, 0.89],
        [0.5, 0.90, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.87, 0.85],
        [0.5, 0.90, 0.94, 0.93, 0.92, 0.91, 0.89, 0.86, 0.85, 0, 0],
        [0.5, 0.90, 0.92, 0.91, 0.90, 0.97, 0.85, 0.83, 0, 0, 0],
        [0.5, 0.90, 0.90, 0.89, 0.87, 0.83, 0.82, 0, 0, 0, 0],
        [0.5, 0.89, 0.89, 0.86, 0.82, 0.81, 0, 0, 0, 0, 0],
        [0.5, 0.86, 0.86, 0.83, 0.80, 0, 0, 0, 0, 0, 0],
        [0.5, 0.85, 0.82, 0.80, 0.78, 0, 0, 0, 0, 0, 0],
        [0.5, 0.80, 0.80, 0.76, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 0.74, 0.75, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    dcac_convert_loss = 0.94
    rad_per_sec_to_rev_per_min = 9.5492966 #avToRPM
    number_of_cells = 192