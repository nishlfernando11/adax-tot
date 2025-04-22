import requests
import numpy as np

#print time
from datetime import datetime
import pylsl
from pylsl import StreamInlet, resolve_byprop
import time
import json

# Generate dummy EEG data with 32 channels and 768 time steps per channel
# eeg_data = np.random.randn(32, 768)  # 32 channels, each with 768 time steps

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

from scipy.signal import butter, lfilter, iirnotch

# Bandpass Filter
def butter_bandpass(lowcut, highcut, fs=256.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(data, lowcut, highcut, fs=256.0):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

# Notch Filter
def apply_notch_filter(data, freq=50.0, fs=256.0, Q=30):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    return lfilter(b, a, data)


def clean_ecg_data(ecg_data, fs=128.0):
    """
    ecg_data: np.array of shape (2, N) with [lead_one_mv, lead_two_mv]
    """
    ecg_data = np.nan_to_num(ecg_data, nan=0.0, posinf=0.0, neginf=0.0)

    cleaned = []
    for ch in ecg_data:
        ch = apply_notch_filter(ch, fs=fs)              # 50Hz removal
        ch = apply_bandpass_filter(ch, fs=fs)           # 0.5–40 Hz bandpass
        cleaned.append(ch)

    return np.array(cleaned)

# === ECG Cleaning ===
def clean_ecg_data(ecg_data):
    """
    ecg_data: np.array of shape (2, N) where 2 channels = [lead_one_mv, lead_two_mv]
    """
    # Step 1: Clean NaNs/Infs
    ecg_data = np.nan_to_num(ecg_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 2: Apply filtering to each channel
    cleaned = []
    for ch in ecg_data:
        ch = apply_notch_filter(ch)          # Remove 50Hz powerline noise
        ch = apply_bandpass_filter(ch)       # ECG range: 0.5–40Hz
        cleaned.append(ch)
    ecg_data = np.array(cleaned)

    # Step 3: Optional Z-score clipping for outlier suppression
    z_scores = (ecg_data - np.mean(ecg_data, axis=1, keepdims=True)) / \
               (np.std(ecg_data, axis=1, keepdims=True) + 1e-8)
    ecg_data = np.where(np.abs(z_scores) > 5,
                        np.sign(z_scores) * 5 * np.std(ecg_data, axis=1, keepdims=True),
                        ecg_data)

    print("ECG data shape after filtering:", ecg_data.shape)
    return ecg_data

import numpy as np
import time
import requests
from datetime import datetime

REQUIRED_LEN = 256  # or whatever your model expects

def fix_signal_length(arr, target_len=REQUIRED_LEN):
    if arr.shape[1] > target_len:
        return arr[:, -target_len:]  # Trim from end
    # elif arr.shape[1] < target_len:
    #     pad_width = target_len - arr.shape[1]
    #     return np.pad(arr, ((0, 0), (0, pad_width)), mode='constant')
    return arr

def get_prediction(eeg_data=None, ecg_data=None):
    start_time = datetime.now()

    # Convert to NumPy arrays if provided and not already arrays
    if eeg_data is not None and not isinstance(eeg_data, np.ndarray):
        eeg_data = np.array(eeg_data)
    if ecg_data is not None and not isinstance(ecg_data, np.ndarray):
        ecg_data = np.array(ecg_data)

    # Return if both are missing or empty
    if (eeg_data is None or eeg_data.size == 0) and (ecg_data is None or ecg_data.size == 0):
        print("❌ No data available for prediction.")
        return {}

    # Clean NaNs/Infs
    if eeg_data is not None and eeg_data.size > 0:
        eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
        eeg_cleaned = []
        for ch in eeg_data:
            ch = apply_notch_filter(ch, fs=256.0)
            ch = apply_bandpass_filter(ch, lowcut=1.0, highcut=40.0, fs=256.0)
            eeg_cleaned.append(ch)
        eeg_data = np.array(eeg_cleaned)
        print("✅ EEG shape after filtering:", eeg_data.shape)

        z_scores = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / \
                   (np.std(eeg_data, axis=1, keepdims=True) + 1e-8)
        eeg_data = np.where(np.abs(z_scores) > 5,
                            np.sign(z_scores) * 5 * np.std(eeg_data, axis=1, keepdims=True),
                            eeg_data)
        normalized_eeg_data = (
            eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
        ) / (np.std(eeg_data, axis=1, keepdims=True) + 1e-8)
    else:
        normalized_eeg_data = None

    if ecg_data is not None and ecg_data.size > 0:
        ecg_data = np.nan_to_num(ecg_data, nan=0.0, posinf=0.0, neginf=0.0)
        ecg_cleaned = []
        for ch in ecg_data:
            ch = apply_notch_filter(ch, fs=256.0)
            ch = apply_bandpass_filter(ch, lowcut=0.5, highcut=40.0, fs=256.0)
            ecg_cleaned.append(ch)
        ecg_data = np.array(ecg_cleaned)
        print("✅ ECG shape after filtering:", ecg_data.shape)
    else:
        ecg_data = None

    # Prepare the data dictionary based on what's available
    data = {'eeg_data': None, 'ecg_data': None}
    if normalized_eeg_data is not None:
        eeg_data = fix_signal_length(eeg_data, target_len=256)
        print("EEG data shape after trimming:", eeg_data.shape)
        data["eeg_data"] = normalized_eeg_data.tolist()
    if ecg_data is not None:
        ecg_data = fix_signal_length(ecg_data, target_len=256)
        print("ECG data shape after trimming:", ecg_data.shape)        
        data["ecg_data"] = ecg_data.tolist()

    print("📤 Sending data to API:", list(data.keys()))
    url = "http://localhost:3003/predict/"
    try:
        response = requests.post(url, json=data)
    except Exception as e:
        print(f"❌ Error sending request: {e}")
        return {}

    print("Response status code:", response.status_code)
    print("Response content:", response.content)

    if response.status_code == 200:
        response_data = response.json()
        print("Parsed Response:", response_data)

        response_data["timestamp"] = time.time()
        end_time = datetime.now()
        print("End Time:", end_time)
        print(f"Total Lag: {(end_time - start_time).total_seconds()} seconds\n\n")
        return response_data.copy()

    else:
        print("Response is invalid")
        end_time = datetime.now()
        print("End Time:", end_time)
        print(f"Total Lag: {(end_time - start_time).total_seconds()} seconds\n\n")
        return {}


def get_prediction_old(eeg_data=[], ecg_data=[]):
    start_time = datetime.now()
    
    # Ensure inputs are NumPy arrays
    if eeg_data is not None and not isinstance(eeg_data, np.ndarray):
        eeg_data = np.array(eeg_data)
    if ecg_data is not None and not isinstance(ecg_data, np.ndarray):
        ecg_data = np.array(ecg_data)

    # Now safe to check .size
    if eeg_data is None or eeg_data.size == 0 or ecg_data is None or ecg_data.size == 0:
        print("❌ No data available for prediction.")
        return {}
    


    # Clean NaNs/Infs
    eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
    ecg_data = np.nan_to_num(ecg_data, nan=0.0, posinf=0.0, neginf=0.0)

    # === EEG Preprocessing ===
    eeg_cleaned = []
    for ch in eeg_data:
        ch = apply_notch_filter(ch, fs=256.0)
        ch = apply_bandpass_filter(ch, lowcut=1.0, highcut=40.0, fs=256.0)
        eeg_cleaned.append(ch)
    eeg_data = np.array(eeg_cleaned)
    print("✅ EEG shape after filtering:", eeg_data.shape)

    # Z-score Clipping (optional)
    z_scores = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / \
            (np.std(eeg_data, axis=1, keepdims=True) + 1e-8)
    eeg_data = np.where(np.abs(z_scores) > 5,
                        np.sign(z_scores) * 5 * np.std(eeg_data, axis=1, keepdims=True),
                        eeg_data)

    # EEG Normalization
    normalized_eeg_data = (
        eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
    ) / (np.std(eeg_data, axis=1, keepdims=True) + 1e-8)

    # Trim EEG to last 256 samples
    # normalized_eeg_data = normalized_eeg_data[:, -256:]

    # === ECG Preprocessing ===
    ecg_cleaned = []
    for ch in ecg_data:
        ch = apply_notch_filter(ch, fs=256.0)
        ch = apply_bandpass_filter(ch, lowcut=0.5, highcut=40.0, fs=256.0)
        ecg_cleaned.append(ch)
    ecg_data = np.array(ecg_cleaned)
    print("✅ ECG shape after filtering:", ecg_data.shape)

    # Trim ECG to last 256 samples to match window
    # ecg_data = ecg_data[:, -256:]

    # === Model Input Packaging ===
    # You can now pass these to your model
    # model_input = {
        # "eeg": normalized_eeg_data,  # shape (32, 256)
        # "ecg": ecg_data              # shape (2, 256)
    # }
    
     # Send to FastAPI
    url = "http://localhost:3003/predict/"
    data = {"eeg_data": normalized_eeg_data.tolist(), "ecg_data": ecg_data.tolist()}
    response = requests.post(url, json=data)
    print("Response status code:", response.status_code)
    print("Response content:", response.content)
    # Check if the response is valid
    if response.status_code == 200:
        print("Response is valid")
        response_data = response.json()
        print("Parsed Response:", response_data)

        # Add timestamp
        response_data["timestamp"] = time.time()

        end_time = datetime.now()
        print("End Time:", end_time)
        print(f"Total Lag: {(end_time - start_time).total_seconds()} seconds\n\n")

        model_output = response_data.copy()  # ✅ Now safe to copy
        return model_output

    else:
        print("Response is invalid")
        end_time = datetime.now()
        print("End Time:", end_time)
        print(f"Total Lag: {(end_time - start_time).total_seconds()} seconds\n\n")
        return {}
    
    
    
def get_prediction_older(eeg_data=None, ecg_data=None):
    start_time = datetime.now()
    print("Prediction Start Time:", start_time)
    if not eeg_data or not ecg_data:
        print("No data available for prediction.")
        return {}

    # Clean NaNs/Infs
    eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Clean NaNs/Infs
    ecg_data = np.nan_to_num(ecg_data, nan=0.0, posinf=0.0, neginf=0.0)


    # Apply notch + bandpass filter per channel
    cleaned = []
    for ch in eeg_data:
        ch = apply_notch_filter(ch)         # remove powerline noise (50Hz)
        ch = apply_bandpass_filter(ch)      # bandpass between 1–40Hz
        cleaned.append(ch)
    eeg_data = np.array(cleaned)
    print("EEG data shape after filtering:", eeg_data.shape)
    # Z-score clip (optional fast artifact suppressor)
    z_scores = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / (np.std(eeg_data, axis=1, keepdims=True) + 1e-8)
    eeg_data = np.where(np.abs(z_scores) > 5,
                        np.sign(z_scores) * 5 * np.std(eeg_data, axis=1, keepdims=True),
                        eeg_data)

    # Normalize
    epsilon = 1e-8
    normalized_eeg_data = (
        eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
    ) / (np.std(eeg_data, axis=1, keepdims=True) + epsilon)


    # FIX: Trim to last 256 samples to match model input shape
    normalized_eeg_data = normalized_eeg_data[:, -256:]

    # Send to FastAPI
    url = "http://localhost:3003/predict/"
    data = {"eeg_data": normalized_eeg_data.tolist()}
    response = requests.post(url, json=data)
    print("Response status code:", response.status_code)
    print("Response content:", response.content)
    # Check if the response is valid
    if response.status_code == 200:
        print("Response is valid")
        response_data = response.json()
        print("Parsed Response:", response_data)

        # Add timestamp
        response_data["timestamp"] = time.time()

        end_time = datetime.now()
        print("End Time:", end_time)
        print(f"Total Lag: {(end_time - start_time).total_seconds()} seconds\n\n")

        model_output = response_data.copy()  # ✅ Now safe to copy
        return model_output

    else:
        print("Response is invalid")
        end_time = datetime.now()
        print("End Time:", end_time)
        print(f"Total Lag: {(end_time - start_time).total_seconds()} seconds\n\n")
        return {}
    # model_output = {
    #     "timestamp": datetime.now().isoformat(),
    #     "prediction": "excitement",
    #     "probabilities": {
    #         "excitement": 0.9462,
    #         "stress": 0.00005,
    #         "depression": 0.00071,
    #         "relaxation": 0.0529
    #     }
    # }
    # return model_output


# def get_prediction(eeg_data):
#     """
#     Send EEG data to FastAPI server for prediction.
    
#     Parameters:
#     - eeg_data: numpy array of shape (32, 768)
    
#     Returns:
#     - response: JSON response from the FastAPI server
#     """
#     # Record the start time
#     start_time = datetime.now()
#     print("Start Time: ", start_time)
#     # Normalize the EEG data (mean=0, std=1 across each channel)
#     # normalized_eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / np.std(eeg_data, axis=1, keepdims=True)
    
#     epsilon = 1e-8
#     normalized_eeg_data = (
#         eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
#     ) / (np.std(eeg_data, axis=1, keepdims=True) + epsilon)

#     # Send the normalized EEG data to the FastAPI server
#     url = "http://localhost:3003/predict/"
#     data = {
#         "eeg_data": normalized_eeg_data.tolist()  # Convert to list for JSON compatibility
#     }

#     response = requests.post(url, json=data)
#     print(response.json())  # Prints model prediction
#     end_time = datetime.now()
#     print("End Time: ", end_time)



#     # Calculate the lag in seconds and milliseconds
#     lag_seconds = (end_time - start_time).total_seconds()  # Difference in seconds

#     # Print the lag in seconds and milliseconds
#     print(f"Total Lag: {lag_seconds} seconds\n\n")
#     return response.json()  # Prints model prediction


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


# def fix_timestamp(timestamp):
#     if timestamp is None:
#         return None
#     return unix_start_time + (timestamp - lsl_start_time)

def fix_timestamp(timestamp):
    if timestamp is None:
        return None
    return time.time() + (timestamp - pylsl.local_clock())

# if __name__ == "__main__":
    
#     eeg_inlet = safe_resolve('Emotiv_EEG')
#     print("eeg_inlet", eeg_inlet)

#     lsl_start_time = pylsl.local_clock()
#     unix_start_time = time.time()
#     lsl_to_unix_offset = unix_start_time - lsl_start_time
#     eeg_offset = safe_time_correction(eeg_inlet, "Emotiv_EEG")
#     # buffer = []
#     # if eeg_inlet: 
#     #     eeg_sample, eeg_timestamp = eeg_inlet.pull_sample(timeout=0.0)
#     #     if eeg_sample:
#     #         print_data(eeg_sample, eeg_timestamp, eeg_offset, "Emotiv_EEG")
#     #         channels = eeg_sample['data'][2:34]  # Assuming the first two values and last 4 values are not EEG data
#     #         buffer.append(channels)
#     buffer = np.zeros((32, 0))  # shape: (channels, time_steps)
#     is_data_available = True  # Flag to check if data is available
#     while is_data_available:
#         # print("Buffer shape:", buffer.shape)
#         # print("Buffer size:", buffer.size)
#         # print("Buffer length:", len(buffer))
#         if buffer.shape[1] == 256*5:
#             copy_buffer = buffer.copy()
#             buffer = np.zeros((32, 0))  # Reset buffer after processing
#             print("Buffer shape after reset:", buffer.shape)
#             print("Buffer shape:", copy_buffer.shape)
#             prediction = get_prediction(copy_buffer
                                        
#                                         )
#             print("\n==============\nPrediction:", prediction)
#         eeg_sample, eeg_timestamp = eeg_inlet.pull_sample(timeout=1.0)
#         if eeg_sample:
#             is_data_available = True  # Data is available
#             print_data(eeg_sample, eeg_timestamp, eeg_offset, "Emotiv_EEG")
#             # print("eeg_sample", eeg_sample)
#             json_sample = json.loads(eeg_sample[0])
#             new_sample = json_sample['data'][2:34]  # Assuming the first two values and last 4 values are not EEG data
#             new_sample = np.array(new_sample)
#             new_sample = new_sample.reshape(32, 1)  # column vector
#             buffer = np.hstack((buffer, new_sample))  # append along time axis
            
#         else:
#             is_data_available = False
#             print("No data available, waiting...") 
            
    # Generate dummy EEG data with 32 channels and 768 time steps per channel
    # eeg_data = np.random.randn(32, 768)  # 32 channels, each with 768 time steps

        
    # prediction = get_prediction(eeg_data)
    # print("Prediction:", prediction)