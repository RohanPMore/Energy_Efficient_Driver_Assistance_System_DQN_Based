import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import os

# Import the smooth_curve_function from the separate file
from functions.curvature.smooth_curve import smooth_curve_function

def curvature_function(track_data, delta_s):
    # Extract columns from track_data
    Allowed_Speed_raw = track_data[:, 5]  # In kmph
    Allowed_Speed = Allowed_Speed_raw / 3.6  # In m/s
    posX, posY, posZ = smooth_curve_function(track_data[:, 2], track_data[:, 1], track_data[:, 3])

    distanceRawtemp = track_data[:, 4]
    if distanceRawtemp[0] == 0:
        distanceRaw = distanceRawtemp
    else:
        distanceRaw = distanceRawtemp - distanceRawtemp[0]

    # For Curvature based on Bezier Curve Quadratic
    deltaS = np.zeros(track_data.shape[0])
    curvature = np.zeros(track_data.shape[0])
    for j in range(track_data.shape[0] - 5):
        deltaS[j] = distanceRaw[j]
        px1 = 2 * (posX[j+1] - posX[j])
        px2 = 2 * (posX[j+2] - posX[j+1])
        dpx = px2 - px1
        py1 = 2 * (posY[j+1] - posY[j])
        py2 = 2 * (posY[j+2] - posY[j+1])
        dpy = py2 - py1
        k = np.zeros(11)
        for t in range(11):
            dt = t / 10.0
            dx = px1 * (1 - dt) + px2 * dt
            dy = py1 * (1 - dt) + py2 * dt
            k[t] = (dx * dpy - dy * dpx) / ((dx**2 + dy**2)**(3/2))
        curvature[j] = abs(np.mean(k))
    
    #print(curvature)

    # For Interpolated Values of Distance and curvature
    v = Allowed_Speed[:-1]
    deltaS_u, unique_indices = np.unique(deltaS, return_index=True)
    v_u = v[unique_indices]
    curvature_u = curvature[unique_indices]
    dInterval = delta_s  # Distance interval

    distance = []
    interpl_curvature = []
    interpl_velspeed_ms = []
    interpl_posZ = []

    for u in range(0, int(max(deltaS)), dInterval):
        j = u // dInterval
        distance.append(u)
        interpl_curvature.append(interp1d(deltaS_u, curvature_u, kind='linear')(u))
        interpl_velspeed_ms.append(interp1d(deltaS_u, v_u, kind='linear')(u))
        interpl_posZ.append(interp1d(distanceRaw, posZ, kind='linear')(u))

    distance = np.array(distance)
    interpl_curvature = np.array(interpl_curvature)
    interpl_velspeed_ms = np.array(interpl_velspeed_ms)
    interpl_posZ = np.array(interpl_posZ)

    if np.sum(posZ) == 0:
        interpl_posZ.fill(0)

    # Ensure the window length is appropriate for the size of interpl_curvature
    window_length = min(21, len(interpl_curvature) // 2 * 2 + 1)  # Ensure window length is odd and less than or equal to array size
    if window_length < 3:
        window_length = 3  # Minimum valid window length for savgol_filter

    # Using Savitzky-Golay filter to smooth the curvature (equivalent to MATLAB smooth function)
    interpl_curvature = savgol_filter(interpl_curvature, window_length=window_length, polyorder=2)
    interpl_curvature[-5:] = 0  # Final condition

    req_speed_ms = np.zeros_like(interpl_curvature)
    for u in range(len(interpl_curvature)):
        if interpl_curvature[u] != 0:
            req_speed_ms[u] = np.sqrt(9.81 / interpl_curvature[u])
        else:
            req_speed_ms[u] = interpl_velspeed_ms[u]
        
        # Handle NaN values by replacing with the previous value
        if np.isnan(req_speed_ms[u]) and u > 0:
            req_speed_ms[u] = req_speed_ms[u - 1]
        elif np.isnan(req_speed_ms[u]) and u == 0:
            req_speed_ms[u] = 0  # Assuming 0 as the default value if the first value is NaN


    req_speed = req_speed_ms * 3.6  # Conversion to kmph
    interpl_velspeed = interpl_velspeed_ms * 3.6  # Conversion to kmph

    curvature_speed = np.minimum(req_speed, interpl_velspeed)

    # Save the results to the results folder
    Results_folder = os.path.join(os.getcwd(), 'Results')
    if not os.path.exists(Results_folder):
        os.makedirs(Results_folder)

    return distance, curvature_speed, interpl_velspeed_ms, interpl_posZ

# Example usage:
#if __name__ == "__main__":
    # Example track_data: [time, y, x, z, distance, speed]
    #track_data = np.array([
        #[0, 1, 1, 1, 0, 50],
        #[1, 2, 2, 2, 1, 50],
        #[2, 2, 2, 2, 2, 50],
        #[3, 3, 3, 3, 3, 50],
        #[4, 4, 4, 4, 4, 50],
        #[5, 5, 5, 5, 5, 50]
    #])
    #delta_s = 1
    #distance, curvature_speed, interpl_velspeed_ms, interpl_posZ = curvature_function(track_data, delta_s)
    #print(distance)
    #print(curvature_speed)
    #print(interpl_velspeed_ms)
    #print(interpl_posZ)
