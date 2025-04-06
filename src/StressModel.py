import requests
import numpy as np

#print time
from datetime import datetime
import pylsl
from pylsl import StreamInlet, resolve_byprop
import time
import json

# Generate dummy EEG data with 32 channels and 768 time steps per channel
eeg_data = np.random.randn(32, 768)  # 32 channels, each with 768 time steps

def get_stress_pred(eeg_data):
    # Record the start time
    start_time = datetime.now()
    print("Start Time: ", start_time)
    # Normalize the EEG data (mean=0, std=1 across each channel)
    normalized_eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / np.std(eeg_data, axis=1, keepdims=True)

    # Send the normalized EEG data to the FastAPI server
    url = "http://localhost:3003/predict/"
    data = {
        "eeg_data": normalized_eeg_data.tolist()  # Convert to list for JSON compatibility
    }

    response = requests.post(url, json=data)
    if response:
        response = response.json()
    print(response)  # Prints model prediction
    end_time = datetime.now()
    print("End Time: ", end_time)

    # Calculate the lag in seconds and milliseconds
    lag_seconds = (end_time - start_time).total_seconds()  # Difference in seconds

    # Print the lag in seconds and milliseconds
    print(f"Total Lag: {lag_seconds} seconds")
    return response



import requests
import numpy as np

#print time
from datetime import datetime

def get_prediction(eeg_data):
    """
    Send EEG data to FastAPI server for prediction.
    
    Parameters:
    - eeg_data: numpy array of shape (32, 768)
    
    Returns:
    - response: JSON response from the FastAPI server
    """
    # Record the start time
    start_time = datetime.now()
    print("Start Time: ", start_time)
    # Normalize the EEG data (mean=0, std=1 across each channel)
    # normalized_eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / np.std(eeg_data, axis=1, keepdims=True)
    
    epsilon = 1e-8
    normalized_eeg_data = (
        eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
    ) / (np.std(eeg_data, axis=1, keepdims=True) + epsilon)

    # Send the normalized EEG data to the FastAPI server
    url = "http://localhost:3003/predict/"
    data = {
        "eeg_data": normalized_eeg_data.tolist()  # Convert to list for JSON compatibility
    }

    response = requests.post(url, json=data)
    print(response.json())  # Prints model prediction
    end_time = datetime.now()
    print("End Time: ", end_time)



    # Calculate the lag in seconds and milliseconds
    lag_seconds = (end_time - start_time).total_seconds()  # Difference in seconds

    # Print the lag in seconds and milliseconds
    print(f"Total Lag: {lag_seconds} seconds\n\n")
    return response.json()  # Prints model prediction


def safe_resolve(name):
    try:
        print(f"Resolving stream {name}...")
        streams = resolve_byprop('name', name, timeout=2)
        if not streams:
            print(f"Stream {name} not found.")
            return None
        return StreamInlet(streams[0]) if streams else None
    except Exception as e:
        print(f"Error resolving stream {name}: {e}")
        return None

def safe_time_correction(inlet, name):
    try:
        if inlet:
            return inlet.time_correction()
        print(f"Warning: {name} stream not available.")
        return 0.0
    except Exception as e:  
        print(f"Error getting time correction for {name}: {e}")
        return 0.0

def print_data(data_sample, data_timestamp, data_offset, type):
    # print(f"---{type} sample: ", data_sample)
    # print(f"---{type} timestamp: ", data_timestamp)
    sample = str(data_sample[0])
    data_sample = json.loads(sample)
    cor_data_timestamp = data_timestamp + data_offset
    # print(f"{type}_sample ", data_sample)
    # print(f"---{type} timestamp: {cor_data_timestamp}")


def fix_timestamp(timestamp):
    if timestamp is None:
        return None
    return unix_start_time + (timestamp - lsl_start_time)


if __name__ == "__main__":
    
    eeg_inlet = safe_resolve('Emotiv_EEG')
    print("eeg_inlet", eeg_inlet)

    lsl_start_time = pylsl.local_clock()
    unix_start_time = time.time()
    lsl_to_unix_offset = unix_start_time - lsl_start_time
    eeg_offset = safe_time_correction(eeg_inlet, "Emotiv_EEG")
    # buffer = []
    # if eeg_inlet: 
    #     eeg_sample, eeg_timestamp = eeg_inlet.pull_sample(timeout=0.0)
    #     if eeg_sample:
    #         print_data(eeg_sample, eeg_timestamp, eeg_offset, "Emotiv_EEG")
    #         channels = eeg_sample['data'][2:34]  # Assuming the first two values and last 4 values are not EEG data
    #         buffer.append(channels)
    buffer = np.zeros((32, 0))  # shape: (channels, time_steps)
    is_data_available = True  # Flag to check if data is available
    while is_data_available:
        # print("Buffer shape:", buffer.shape)
        # print("Buffer size:", buffer.size)
        # print("Buffer length:", len(buffer))
        if buffer.shape[1] == 256*5:
            copy_buffer = buffer.copy()
            buffer = np.zeros((32, 0))  # Reset buffer after processing
            print("Buffer shape after reset:", buffer.shape)
            print("Buffer shape:", copy_buffer.shape)
            prediction = get_prediction(copy_buffer
                                        
                                        )
            print("\n==============\nPrediction:", prediction)
        eeg_sample, eeg_timestamp = eeg_inlet.pull_sample(timeout=1.0)
        if eeg_sample:
            is_data_available = True  # Data is available
            print_data(eeg_sample, eeg_timestamp, eeg_offset, "Emotiv_EEG")
            # print("eeg_sample", eeg_sample)
            json_sample = json.loads(eeg_sample[0])
            new_sample = json_sample['data'][2:34]  # Assuming the first two values and last 4 values are not EEG data
            new_sample = np.array(new_sample)
            new_sample = new_sample.reshape(32, 1)  # column vector
            buffer = np.hstack((buffer, new_sample))  # append along time axis
            
        else:
            is_data_available = False
            print("No data available, waiting...") 
            
    # Generate dummy EEG data with 32 channels and 768 time steps per channel
    # eeg_data = np.random.randn(32, 768)  # 32 channels, each with 768 time steps

        
    # prediction = get_prediction(eeg_data)
    # print("Prediction:", prediction)