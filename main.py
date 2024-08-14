# main.py
# Import libraries
import os
import glob
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.signal import butter, filtfilt
from math import atan

# Import functions
from functions.curvature.curvature_IPG import curvature_function
from functions.mergeStructsRecursive import mergeStructsRecursive_function
from functions.rl.main_dqn import main_dqn_function

# functions
def get_track_data_files():
    folder_path = './data/route_data/'
    track_data_files = glob.glob(os.path.join(folder_path, '*.csv'))
    return track_data_files

def get_result_folder():
    timestamp_format = "%Y_%m_%d__%H_%M_%S"
    timestamp = datetime.now().strftime(timestamp_format)
    result_location = os.path.join(os.getcwd(), 'results', timestamp)
    if not os.path.exists(result_location):
        os.makedirs(result_location)
    return result_location

def process_track_data(track_data_files, options):
    track_data = [None] * len(track_data_files)
    
    for i, track_data_file in enumerate(track_data_files):
        options_track = options.copy()
        data_location = track_data_file
        track_name = os.path.basename(track_data_file).replace('.csv', '')
        
        single_track_data = pd.read_csv(data_location)
        track_data[i] = process_single_track(single_track_data, options_track, track_name)

        # Save single_track_data to Excel
        single_track_data.to_excel(os.path.join(options['result_location'], f'single_track_data_{track_name}.xlsx'), index=False)
    
    # Save track_parts to Excel
    #for i, track in enumerate(track_data):
        #if track is not None:
            #for j, part in enumerate(track):
                #part_df = part['track_data']
                #if isinstance(part_df, pd.DataFrame):
                    #part_df.to_excel(os.path.join(options['result_location'], f'track_parts_{i+1}_{j+1}.xlsx'), index=False)

def process_single_track(track_data, options, track_name):
    track_parts = []
    
    # Split track into parts depending on stop sign position
    track_parts = recursive_split(track_parts, track_data, False)
    #print(f'Initial track_parts for {track_name}:')
    #for idx, part in enumerate(track_parts):
        #print(f'Part {idx}:', part)
    single_track = process_track_parts(track_parts, options, track_name)
    
    return single_track

def recursive_split(segments, input_array, start_flag):
    if input_array.empty:
        return segments

    zero_index = input_array[input_array['speedlimit'] == 0].index.min()

    if pd.isna(zero_index):
        segments.append(process_segment(input_array, start_flag, False))
    else:
        segment = input_array.loc[:zero_index]
        segments.append(process_segment(segment, start_flag, True))
        remaining_array = input_array.loc[zero_index + 1:]
        segments = recursive_split(segments, remaining_array, True)

    return segments

def process_segment(segment, start_flag, stop_flag):
    if stop_flag:
        segment.at[segment.index[-1], 'speedlimit'] = segment['speedlimit'].iloc[-2]
    return {'track_data': segment, 'stop_flag': stop_flag, 'start_flag': start_flag}

def process_track_parts(track_parts, options, track_name):
    for j in range(len(track_parts)):
        track_parts[j]['track_data'] = process_track_part(track_parts[j], options, track_name)
        #print(f'Processed track_parts[{j}][track_data] for {track_name}:')
        #print(track_parts[j]['track_data'])
    return track_parts

def process_track_part(track_part_dict, options, track_name):
    track_part, options = preprocess_track_part(track_part_dict, options)
    # Add the lines from the MATLAB code
    track_part['result'] = main_dqn_function(track_part['track_data_interp'], options)
    #export_weights(beta, lambda_, options['result_location'], options['weight_file_name'], track_name)
    return track_part

def preprocess_track_part(track_part_dict, options):
    stop_flag = track_part_dict['stop_flag']
    start_flag = track_part_dict['start_flag']
    options['stop_flag'] = stop_flag
    options['start_flag'] = start_flag
    track_part = track_part_dict['track_data']
    distance, curvature_speed, interpl_velspeed_ms, interpl_posZ = curvature_function(track_part.to_numpy(), options['delta_s'])
    
    v_max_curve = curvature_speed / 3.6
    v_max = interpl_velspeed_ms
    v_min = np.zeros(len(v_max_curve))
    #print('interpl_posZ')
    #print(interpl_posZ)
    
    alpha_filt, slope = get_slope(interpl_posZ, options['delta_s'])
    #print('alpha_filt')
    #print(alpha_filt)
    #print('slope')
    #print(slope)
    
    if stop_flag:
        v_min[-2:] = [0, 0]
        v_max[-2:] = [v_max[-2] / 2, 2]
        v_max_curve[-2:] = [v_max_curve[-2] / 2, 2]
    
    track_part_dict['track_data_interp'] = {
        'distance': distance,
        'distance_next': distance[1:],
        'alpha': alpha_filt, #change
        'slope': slope, #change
        'v_min': v_min,
        'v_max': v_max,
        'v_max_curve': v_max_curve
    }
    
    return track_part_dict, options

def get_slope(z, delta_s):
    track_length = len(z)
    alpha = np.zeros(track_length)

    for j in range(track_length - 1):
        alpha[j] = atan((z[j + 1] - z[j]) / delta_s)
    
    alpha[-1] = alpha[-2]  # Ensure last element is the same as the second last

    if delta_s <= 5:
        half_power_frequency = 0.3 / 5 * delta_s
        b, a = butter(1, half_power_frequency, btype='low', analog=False)
        alpha_filt = filtfilt(b, a, alpha)
    else:
        alpha_filt = alpha

    slope = np.zeros(track_length)
    for j in range(track_length):
        if j + int(100 / delta_s) >= track_length:
            slope[j] = (z[-1] - z[j]) / 100.0
        else:
            slope[j] = (z[j + int(100 / delta_s)] - z[j]) / 100.0
    
    return alpha_filt, slope

def auto_tune_velocity_optimization(track_data_interp, options):
    pass

def export_weights(beta, lambda_, result_location, weight_file_name, track_name):
    pass

def merge_track_parts(j, track_parts):
    pass

def merge_tracks(final_track_data, final_track_data_interp, final_result_data, single_track):
    pass

def export_result(final_track_data, final_track_data_interp, final_result_data, result_location, filename):
    pass

def main():
    options = {
        'start_flag': 1,
        'stop_flag': 0,
        'delta_s': 5,
        'result_location': get_result_folder()
    }

    #print("Options dictionary:")
    #print(options)

    #print(dqn_function())
    #print(mergeStructsRecursive_function())

    track_data_files = get_track_data_files()
    process_track_data(track_data_files, options)
    
    #print("Track Data Files:")
    #for file in track_data_files:
        #print(file)

if __name__ == "__main__":
    main()
