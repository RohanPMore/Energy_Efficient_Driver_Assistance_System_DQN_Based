import numpy as np
from scipy.signal import savgol_filter

def smooth_curve_function(xData, yData, zData):
    def linear_smooth(data):
        newData = np.zeros(len(data))  # Initialize new data array with zeros
        i = 0
        while i < len(data):
            j = i
            # Increment j until a different value from data[i] is found or end of array is reached
            while j < len(data) - 1:
                j += 1
                if data[j] != data[i]:
                    break

            if j == i:
                # If no different value was found, assign the same value to newData and move to the next element
                newData[i] = data[i]
                i += 1
                continue

            if j < len(data):
                slope = 1.0 / (j - i)  # Calculate the slope
                delta = data[j] - data[i]  # Calculate the difference

                # Linearly interpolate the values
                for k in range(i, j):
                    newData[k] = data[k] + (slope * (k - i) * delta)
                newData[j] = data[j]  # Assign the value at position j
            i = j  # Move i to j to continue the process

        return newData

    # Linear smoothing for x, y, and z Data
    smooth_xData = linear_smooth(xData)
    smooth_yData = linear_smooth(yData)
    smooth_zData = linear_smooth(zData)

    # Additional smoothing using Savitzky-Golay filter
    def apply_savgol_filter(data):
        window_length = min(21, len(data))
        if window_length % 2 == 0:
            window_length -= 1  # Ensure window_length is odd
        return savgol_filter(data, window_length=window_length, polyorder=2)

    smooth_yData = apply_savgol_filter(smooth_yData)
    smooth_zData = apply_savgol_filter(smooth_zData)

    return smooth_xData, smooth_yData, smooth_zData

# Example usage
#xData = np.array([1, 2, 2, 3, 4])
#yData = np.array([1, 2, 2, 3, 4])
#zData = np.array([1, 2, 2, 3, 4])

#smooth_xData, smooth_yData, smooth_zData = smooth_curve_function(xData, yData, zData)
#print(smooth_xData)
#print(smooth_yData)
#print(smooth_zData)
