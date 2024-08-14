import numpy as np
import math
from scipy.interpolate import interp2d  
#from functions.rl.main_dqn import Constants, CostFunction
from functions.rl.Constant import CostFunction,Constants


# fcnEV.py Energy Model for EV 
# Implement dynamics function to compute next state based on current state and action
# updating velocity, SOC, and time

class fcnEV_Class:
    def __init__(self, prev_state, action, alpha, distance, cost_function):
        
        ##---------------------------------State and Input Initialization-------------------------##
        self.velocity__m_per_s = prev_state[0]  
        self.state_of_charge = prev_state[1]     
        if (len(prev_state) == 2):
            #self.total_time__s =  current_state[2] 
            self.time__s = prev_state[2]  
        else:
            #self.total_time__s =  0 
            self.time__s = 0             
        self.torque_requested__Nm = action
        self.delta_distance__m = distance
        self.road_angle__rad = alpha
        self.cost_function = cost_function
        self.aerodynamic_friction_force__N = 0
        self.rolling_friction_force__N = 0
        self.uphill_driving_force__N = 0
        self.acceleration__m_per_s2 = 0
        self.delta_time_segment__s = 0
        #self.total_time_next__s = 0
        self.time_next__s = 0
        self.total_time_segment_s = 0
        self.infeasible_time = 0
        self.velocity_next__m_per_s = 0
        self.mean_velocity__m_per_s = 0
        self.infeasible_velocity = 0
        self.motor_revolution__rev_per_min = 0
        self.infeasible_motor_revolution = 0
        self.max_motor_torque__Nm = 0
        self.max_negative_motor_torque__Nm = 0
        self.min_negative_motor_torque__Nm = 0
        self.infeasible_torque_limits = 0
        self.actual_motor_torque__Nm = 0
        self.hydraulic_braking_torque__Nm = 0
        self.hydraulic_wheel_torque__Nm = 0
        self.throttle = 0
        self.brake = 0
        self.conversion_efficiency = 0
        self.auxiliary_power_W = 0
        self.hydraulic_braking_power_W = 0
        self.total_electric_power__W = 0
        self.electric_power_battery_cell__W = 0
        self.battery_cell_resistance__ohm = 0
        self.battery_cell_current__A = 0
        self.battery_cell_voltage__V = 0
        self.battery_cell_output_voltage__V = 0
        self.state_of_charge_next = 0
        self.battery_power__W = 0
        self.infeasible_battery = 0
        self.costs = 0

        ##-----------------------------Calculate longitudinal system dynamics----------------------##
        self.aerodynamic_friction_force__N, self.rolling_friction_force__N, self.uphill_driving_force__N, self.acceleration__m_per_s2 = self.calculate_longitudinal_system_dynamics()

        ##-------------------Calculate time duration of segment: delta time segment----------------##
        self.delta_time_segment__s, self.time_next__s, self.infeasible_time = self.calculate_delta_time() 

        ##-------------------------------Update Velocity-------------------------------------------##
        self.velocity_next__m_per_s, self.mean_velocity__m_per_s, self.infeasible_velocity = self.update_vehicle_velocity()
        #print('Init Current State Velocity')
        #print(self.velocity_next__m_per_s)

        ##-------------------------------Calculate Motor Revolution--------------------------------##
        self.motor_revolution__rev_per_min, self.infeasible_motor_revolution = self.calculate_motor_revolution()

        ##-------------------------------Calculate Motor Torque Limits------------------------------##
        self.max_motor_torque__Nm, self.max_negative_motor_torque__Nm, self.min_negative_motor_torque__Nm, self.infeasible_torque_limits = self.calculate_motor_torque_limits()

        ##------------------------------Calculate Torque Distributions------------------------------##
        self.actual_motor_torque__Nm, self.hydraulic_braking_torque__Nm, self.hydraulic_wheel_torque__Nm = self.calculate_torque_distribution()

        ##-----------------------------Calculate Pedal Position (TODO)------------------------------##
        throttle, brake = self.calculate_pedal_positions()

        ##-----------------------------Calculate Required Power ------------------------------------##
        self.conversion_efficiency, self.auxiliary_power_W, self.hydraulic_braking_power_W, self.total_electric_power__W, self.electric_power_battery_cell__W = self.calculate_required_power()
        
        ##-----------------------------Calculate Battery Power--------------------------------------##
        self.battery_cell_resistance__ohm, self.battery_cell_current__A, self.battery_cell_voltage__V, self.battery_cell_output_voltage__V, self.state_of_charge_next, self.battery_power__W, self.infeasible_battery = self.calculate_battery_power()

        ##-----------------------------Calculate Infeasibilties--------------------------------------##
        self.infeasible = self.infeasible_time + self.infeasible_velocity + self.infeasible_motor_revolution + self.infeasible_torque_limits + self.infeasible_battery

        ##-----------------------------Calculate Costs-----------------------------------------------##
        self.energy_consumption, self.total_energy_consumption = self.rewards_energy_efficiency_func()

        ##-----------------------------Calculate next states-----------------------------------------##
        self.next_state, self.total_time_segment_s = self.get_state()
    
    def calculate_longitudinal_system_dynamics(self):
        # model "mechanicle dynamic"
       
        # part 1: aerodynamic friction losses, unit: N
        aerodynamic_friction_force__N = (0.5 * Constants.air_density__kg_per_m3 * Constants.front_area__m2 *Constants.aero_drag_coeff * (self.velocity__m_per_s + Constants.wind_velocity__m_per_s) ** 2)
        # part 2: rolling friction losses, unit: N
        rolling_friction_force__N = (Constants.rolling_friction_resistance_coeff * Constants.gravitational_acceleration__m_per_s2 * Constants.vehicle_weight__kg * math.cos(self.road_angle__rad))
        # part 3: uphill driving force, unit: N
        uphill_driving_force__N = (Constants.vehicle_weight__kg * Constants.gravitational_acceleration__m_per_s2 * math.sin(self.road_angle__rad))
        # acceleration of the vehicle, unit: m/s2
        acceleration__m_per_s2 = ((Constants.gear_ratio * Constants.gear_box_efficiency * self.torque_requested__Nm / Constants.wheel_radius__m - aerodynamic_friction_force__N - rolling_friction_force__N - uphill_driving_force__N) / Constants.total_weight__kg)    
        return aerodynamic_friction_force__N, rolling_friction_force__N, uphill_driving_force__N, acceleration__m_per_s2
    
    def calculate_delta_time(self):
        # The vehicle drives within h (h = inp.Ts) with a constant speed 
        if (self.velocity__m_per_s == 0): # remove NaN and Inf in time_constant_speed
            delta_time_constant_speed__s =0
        else:
            delta_time_constant_speed__s = self.delta_distance__m / self.velocity__m_per_s
        
        # Calculate the discriminant for the constant acceleration case
        final_velocity__m_per_s = self.velocity__m_per_s*self.velocity__m_per_s + 2*self.acceleration__m_per_s2*self.delta_distance__m # v2 = u2 +2as
        if (final_velocity__m_per_s < 0):
            final_velocity__m_per_s = 0
        
        # The vehicle drives within h (h = self.distance) with a constant acceleration
        if (self.acceleration__m_per_s2 == 0):
            delta_time_constant_acceletation__s =0 # remove NaN and Inf in time_constant_acceletation
        else:
            delta_time_constant_acceletation__s = (-self.velocity__m_per_s + math.sqrt(final_velocity__m_per_s))/self.acceleration__m_per_s2 
     
        # Combine time intervals based on acceleration magnitude
        delta_time__s = delta_time_constant_speed__s * (np.abs(self.acceleration__m_per_s2) < Constants.min_acceleration__m_per_s2) + delta_time_constant_acceletation__s * (np.abs(self.acceleration__m_per_s2) >= Constants.min_acceleration__m_per_s2)
        
        # Update total time
        time_next__s = delta_time__s + self.time__s
        
        # Check for infeasibility: delta_time__s <= 0 or discriminant < 0
        infeasible = (delta_time__s <= 0) | (final_velocity__m_per_s < 0) 
        return delta_time__s, time_next__s, infeasible
    
    def update_vehicle_velocity(self):
        # Update speed of the vehicle in dependence on the distance
        self.velocity_next__m_per_s = self.acceleration__m_per_s2*self.delta_time_segment__s + self.velocity__m_per_s
        if(self.velocity_next__m_per_s <= 0):
           self.velocity_next__m_per_s = 0
        mean_velocity__m_per_s = (self.velocity__m_per_s + self.velocity_next__m_per_s)/2 
        if(mean_velocity__m_per_s < 0 or self.velocity_next__m_per_s < 0):
            self.infeasible_velocity = True
        return self.velocity_next__m_per_s, mean_velocity__m_per_s, self.infeasible_velocity

    def calculate_motor_revolution(self):
        motor_revolution__rev_per_min = self.mean_velocity__m_per_s*Constants.rad_per_sec_to_rev_per_min*Constants.gear_ratio/Constants.wheel_radius__m
        
        # model 'efficiency of electric motor'
        # required motor revolution speed must between zero and the max. revolution speed
        infeasible = (motor_revolution__rev_per_min >= Constants.motor_max_revolution__rev_per_min) | (motor_revolution__rev_per_min < 0)
        if (motor_revolution__rev_per_min > Constants.motor_max_revolution__rev_per_min):
            motor_revolution__rev_per_min = Constants.motor_max_revolution__rev_per_min
        if (motor_revolution__rev_per_min < 0):
            motor_revolution__rev_per_min = 0 
        return motor_revolution__rev_per_min, infeasible
    
    def calculate_motor_torque_limits(self):
        # Calculate the max positive and negative torque possible at current rpm 
        max_motor_torque__Nm = np.interp(self.motor_revolution__rev_per_min, Constants.max_motor_revolution_lookup__rev_per_min, Constants.max_motor_torque_lookup___Nm)
        max_negative_motor_torque__Nm = np.interp(self.motor_revolution__rev_per_min, Constants.max_motor_revolution_lookup__rev_per_min, Constants.max_negative_motor_torque_lookup__Nm)
        min_negative_motor_torque__Nm = np.interp(self.motor_revolution__rev_per_min, Constants.min_negative_motor_revolution_lookup__r_min, Constants.min_negative_motor_torque_lookup__Nm)
        
        # In the motor mode the max. torque can only be max_motor_torque__Nm, but in the
        infeasible = (np.abs(self.torque_requested__Nm) > max_motor_torque__Nm) & (self.torque_requested__Nm >= 0)
        # In the generator mode the max torque can be max_negative_motor_torque__Nm + max Braking torque
        infeasible = infeasible | ((np.abs(self.torque_requested__Nm) > np.abs(max_negative_motor_torque__Nm) + np.abs(Constants.max_braking_torque__Nm)) & (self.torque_requested__Nm < 0))
        
        # Due to Pedal mapping of Renault Zoe a Torque between zero and
        # min_negative_motor_torque__Nm is not allowed
        infeasible = infeasible | (self.torque_requested__Nm < 0)*(self.torque_requested__Nm > min_negative_motor_torque__Nm) # Not used in Version 1.0.0

        return max_motor_torque__Nm, max_negative_motor_torque__Nm, min_negative_motor_torque__Nm, infeasible   

    def calculate_torque_distribution(self):
        absolute_motor_torque__Nm = 0
        # 1. Pass input torque if less than positive max torque
        absolute_motor_torque__Nm = absolute_motor_torque__Nm + (np.abs(self.torque_requested__Nm) * (np.abs(self.torque_requested__Nm) <= self.max_motor_torque__Nm) * (self.torque_requested__Nm >= 0))
        # 2. Limit input torque to positive max torque when over it
        absolute_motor_torque__Nm = absolute_motor_torque__Nm + (self.max_motor_torque__Nm*(np.abs(self.torque_requested__Nm) > self.max_motor_torque__Nm)*(self.torque_requested__Nm >= 0))
        # 3. Pass input torque if less than negative max torque & max braking torque (take care of sign!)
        absolute_motor_torque__Nm = absolute_motor_torque__Nm + (np.abs(self.torque_requested__Nm)*(self.torque_requested__Nm >= (self.max_negative_motor_torque__Nm + Constants.max_braking_torque__Nm))*(self.torque_requested__Nm < 0)*(self.motor_revolution__rev_per_min > Constants.motor_min_revolution__rev_per_min))
        # 4. Limit input torque to sum of max negative torque & max braking torque
        absolute_motor_torque__Nm = absolute_motor_torque__Nm + np.abs(self.max_negative_motor_torque__Nm + Constants.max_braking_torque__Nm)*(self.torque_requested__Nm < (self.max_negative_motor_torque__Nm + Constants.max_braking_torque__Nm))*(self.torque_requested__Nm < 0)*(self.motor_revolution__rev_per_min > Constants.motor_min_revolution__rev_per_min)
        # 5. When motor revolution is under minium only hydraulic brake is used
        absolute_motor_torque__Nm = absolute_motor_torque__Nm + np.abs(self.torque_requested__Nm)*(self.torque_requested__Nm >= Constants.max_braking_torque__Nm)*(self.torque_requested__Nm<0)*(self.motor_revolution__rev_per_min <= Constants.motor_min_revolution__rev_per_min)
        # 6. When motor revolution is under minium and requested torque is
        # above maximum hydraulic braking torque, maximum hydraulic braking torque
        absolute_motor_torque__Nm = absolute_motor_torque__Nm + np.abs(Constants.max_braking_torque__Nm)*(self.torque_requested__Nm < Constants.max_braking_torque__Nm)*(self.torque_requested__Nm<0)*(self.motor_revolution__rev_per_min <= Constants.motor_min_revolution__rev_per_min)
        
        # Calculate and store current hydraulic braking torque
        # Hydraulic braking torque is difference between current torque and max regen torque at the instant, 
        # but only when current torque is more than regen torque 
        hydraulic_braking_torque__Nm = (np.abs(absolute_motor_torque__Nm) - np.abs(self.max_negative_motor_torque__Nm))*(self.torque_requested__Nm < 0)*(np.abs(absolute_motor_torque__Nm) > np.abs(self.max_negative_motor_torque__Nm))*(self.motor_revolution__rev_per_min > Constants.motor_min_revolution__rev_per_min)
        # When motor revolution is under minium only hydraulic brake is used
        # (This is considered in the max negative torque curve, but due to
        # interpolation errors, this is considered here
        hydraulic_braking_torque__Nm = hydraulic_braking_torque__Nm + np.abs(self.torque_requested__Nm)*(self.torque_requested__Nm >= Constants.max_braking_torque__Nm)*((self.torque_requested__Nm < 0)*(self.motor_revolution__rev_per_min <= Constants.motor_min_revolution__rev_per_min))
        hydraulic_braking_torque__Nm = hydraulic_braking_torque__Nm + np.abs(Constants.max_braking_torque__Nm)*(self.torque_requested__Nm < Constants.max_braking_torque__Nm)*((self.torque_requested__Nm < 0)*(self.motor_revolution__rev_per_min <= Constants.motor_min_revolution__rev_per_min))

        # Pass the input torque to real torque, if above minimum regen rpm
        # Also, pass the hydraulic torque as real torque (if any), when under min regen rpm 
        actual_motor_torque_with_brake__Nm = absolute_motor_torque__Nm*(self.torque_requested__Nm >= 0) - (absolute_motor_torque__Nm*(self.torque_requested__Nm < 0))*(self.motor_revolution__rev_per_min > Constants.motor_min_revolution__rev_per_min) - (hydraulic_braking_torque__Nm*(self.motor_revolution__rev_per_min <= Constants.motor_min_revolution__rev_per_min))
    
        # Calculate effective real motor torque for further calculations
        # Positive symbol of T_hydraulic ensures that it gets substracted with t_mot_real which is always negative when input torque is negative
        actual_motor_torque__Nm = actual_motor_torque_with_brake__Nm*(self.torque_requested__Nm >= 0) - np.abs(actual_motor_torque_with_brake__Nm + hydraulic_braking_torque__Nm)*(self.torque_requested__Nm < 0)*(actual_motor_torque_with_brake__Nm != 0)

        # Calculate hydraulic braking torque at the wheels
        hydraulic_wheel_torque__Nm = Constants.gear_ratio*hydraulic_braking_torque__Nm*(hydraulic_braking_torque__Nm > 0)
        check = np.abs(absolute_motor_torque__Nm) - np.abs(actual_motor_torque__Nm - hydraulic_braking_torque__Nm)
        if np.any(check):
            print('Torque Distribution Failed')
        
        return actual_motor_torque__Nm, hydraulic_braking_torque__Nm, hydraulic_wheel_torque__Nm
    
    def calculate_pedal_positions(self):
        # Calculate Throttle and Brake Position
        throttle = 0
        brake = 0
        return throttle, brake
    
    def calculate_required_power(self):
        # Calculate auxiliary power
        auxiliary_power_W = (
            Constants.touch_screen_radio_state * Constants.touch_screen_radio_power__W +
            Constants.steering_state * Constants.steering_power__W +
            Constants.roll_stabilizer_state * Constants.roll_stabilizer_power__W +
            Constants.air_cooling_state * Constants.air_cooling_power__W +
            Constants.heating_state * Constants.heating_power__W +
            Constants.motor_cooling_state * Constants.motor_cooling_power__W +
            Constants.low_beam_state * Constants.low_beam_power__W
        )

        # Create the interpolation function for conversion efficiency
        interp_func = interp2d(
            Constants.motor_efficiency_torque_lookup__Nm,
            Constants.motor_efficiency_motor_revolution__rev_per_min,
            Constants.motor_efficiency_lookup,
            kind='linear'  # You can choose 'linear', 'cubic', 'quintic'
        )

        # Evaluate the interpolation function to get conversion efficiency
        conversion_efficiency = interp_func(
            np.abs(self.actual_motor_torque__Nm),
            self.motor_revolution__rev_per_min
        )

        # Since conversion_efficiency will be a 2D array, extract the scalar value
        conversion_efficiency = conversion_efficiency.item()

        # Energy consumption of the hydraulic braking
        hydraulic_braking_power_W = self.hydraulic_wheel_torque__Nm * self.mean_velocity__m_per_s / Constants.wheel_radius__m

        # Calculate the required power based on the direction of torque
        positive_model_correction_factor = 1.0
        negative_model_correction_factor = 1.0

        if self.actual_motor_torque__Nm >= 0:
            total_electric_power__W = (1 / conversion_efficiency) * (
                self.actual_motor_torque__Nm * self.motor_revolution__rev_per_min / Constants.rad_per_sec_to_rev_per_min
            )
        else:
            total_electric_power__W = conversion_efficiency * (
                self.actual_motor_torque__Nm * self.motor_revolution__rev_per_min / Constants.rad_per_sec_to_rev_per_min
            )

        total_electric_power__W = positive_model_correction_factor * (
            (total_electric_power__W + auxiliary_power_W) / Constants.dcac_convert_loss
        ) * (total_electric_power__W >= 0) + (
            (total_electric_power__W + auxiliary_power_W) * Constants.dcac_convert_loss
        ) * negative_model_correction_factor * (total_electric_power__W < 0)

        # Calculate the electric power required per battery cell
        electric_power_battery_cell__W = total_electric_power__W / Constants.number_of_cells

        return conversion_efficiency, auxiliary_power_W, hydraulic_braking_power_W, total_electric_power__W, electric_power_battery_cell__W

    def calculate_battery_power(self):
        # Model "battery"
        state_of_discharge = 1 - self.state_of_charge  # x2: SOC
        
        # Resistance calculation for both charge & discharge cases
        # Ensure that state_of_discharge is treated correctly in interpolation
        if np.isscalar(state_of_discharge):
            battery_cell_resistance__ohm = (
                (self.electric_power_battery_cell__W >= 0) * np.interp(state_of_discharge, Constants.state_of_discharge_lookup, Constants.discharge_resistance_lookup__ohm)
            ) + (
                (self.electric_power_battery_cell__W < 0) * np.interp(state_of_discharge, Constants.state_of_discharge_lookup, Constants.charge_resistance_lookup__ohm)
            )
        else:
            battery_cell_resistance__ohm = (
                (self.electric_power_battery_cell__W >= 0) * np.array([np.interp(sod, Constants.state_of_discharge_lookup, Constants.discharge_resistance_lookup__ohm) for sod in state_of_discharge])
            ) + (
                (self.electric_power_battery_cell__W < 0) * np.array([np.interp(sod, Constants.state_of_discharge_lookup, Constants.charge_resistance_lookup__ohm) for sod in state_of_discharge])
            )

        # Open circuit voltage of the battery cell
        battery_cell_voltage__V = np.interp(self.state_of_charge, Constants.state_of_charge_open_circuit_voltage_lookup, Constants.open_circuit_voltage_lookup__V)

        # Output voltage of the battery cell
        battery_cell_output_voltage__V = (battery_cell_voltage__V + np.sqrt(battery_cell_voltage__V**2 - 4*battery_cell_resistance__ohm*self.electric_power_battery_cell__W)) / 2
        infeasible_5 = ((battery_cell_voltage__V**2) < (4*battery_cell_resistance__ohm*self.electric_power_battery_cell__W))
        
        # Handle infeasible cases
        battery_cell_output_voltage__V = battery_cell_voltage__V*(infeasible_5 == 1)/2 + battery_cell_output_voltage__V*(infeasible_5 == 0)
        
        # Output current of the battery cell
        battery_cell_current__A = (battery_cell_voltage__V - battery_cell_output_voltage__V) / battery_cell_resistance__ohm
        
        self.state_of_charge_next = self.state_of_charge - (battery_cell_current__A * self.delta_time_segment__s) / Constants.reference_capacity__F
        
        # Check for infeasibilities
        infeasible = (
            (np.abs(battery_cell_current__A) > Constants.max_inverter_current__A) +
            (battery_cell_output_voltage__V > Constants.max_voltage__V) + 
            (battery_cell_output_voltage__V < Constants.cutoff_voltage__V)
        )
        
        # Check for overpower conditions
        max_discharge_power__W = np.interp(self.state_of_charge, Constants.state_of_charge_discharge_lookup, Constants.max_discharge_power_lookup__W)
        max_charge_power__W = np.interp(self.state_of_charge, Constants.state_of_charge_charge_lookup, Constants.max_power_charge_lookup__W)

        infeasible += (
            (self.total_electric_power__W > 0) * (self.electric_power_battery_cell__W > max_discharge_power__W) + 
            (self.total_electric_power__W < 0) * (np.abs(self.electric_power_battery_cell__W) > max_charge_power__W)
        )

        battery_power__W = battery_cell_voltage__V * battery_cell_current__A * Constants.number_of_cells
    
        return battery_cell_resistance__ohm, battery_cell_current__A, battery_cell_voltage__V, battery_cell_output_voltage__V, self.state_of_charge_next, battery_power__W, infeasible

    def rewards_energy_efficiency_func(self):
        # Calculate max_time__s and max_energy__Ws
        max_time__s = self.delta_distance__m / Constants.min_velocity_m_per_s
        max_energy__Ws = max(Constants.max_discharge_power_lookup__W) * Constants.number_of_cells * max_time__s
        max_torque__Nm = max(Constants.max_motor_torque_lookup___Nm)

        # Calculate costs based on cost_function
        if self.cost_function == CostFunction.ENERGY_EFFICIENCY:
            rewards = (-1)*(self.battery_power__W * Constants.number_of_cells * self.delta_time_segment__s) / max_energy__Ws + (self.torque_requested__Nm / max_torque__Nm)**2 #* self.weight_lambda
            #print('------ Current Step Rewards ---------------')
            #print(rewards)

        #elif self.cost_function == CostFunction.ENERGY_EFFICIENCY_TIME_MINIMIZATION:
            #costs = (self.battery_power__W * Constants.number_of_cells * self.delta_time_segment__s) / max_energy__Ws + (self.delta_time_segment__s / max_time__s) * self.weight_beta + (self.torque_requested__Nm / max_torque__Nm)**2 * self.weight_lambda

        #elif self.cost_function == CostFunction.TIME_EFFICIENCY:
            #costs = self.weight_beta * (self.delta_time_segment__s / max_time__s)

        #elif self.cost_function == CostFunction.INVERSE_ENERGY_EFFICIENCY:
            #costs = -(self.battery_power__W * Constants.number_of_cells * self.delta_time_segment__s) / max_energy__Ws + (self.torque_requested__Nm / max_torque__Nm)**2 * self.weight_lambda

        #elif self.cost_function == CostFunction.INVERSE_ENERGY_EFFICIENCY_TIME_MINIMIZATION:
            #costs = -(self.battery_power__W * Constants.number_of_cells * self.delta_time_segment__s) / max_energy__Ws + (self.delta_time_segment__s / max_time__s) * self.weight_beta + (self.torque_requested__Nm / max_torque__Nm)**2 * self.weight_lambda

        #else:
            #costs = 0

        return rewards
    
    def get_state(self):
        next_state = [self.velocity_next__m_per_s, self.state_of_charge_next, self.time_next__s]
        total_time_segment_s = self.time_next__s + self.delta_time_segment__s
        #print('------ Current State Velocity ---------------')
        #print('velocity_next__m_per_s: ')
        #print(self.velocity_next__m_per_s)
        #print('state_of_charge_next: ')
        #print(self.state_of_charge_next)
        #print('total_time_next__s: ')
        #print(self.total_time_next__s)
        return next_state, total_time_segment_s




