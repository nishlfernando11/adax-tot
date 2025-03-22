import pylsl
import csv
import time
import ast  # To safely convert stringified lists back to lists

# Define LSL stream properties (matching original Windows settings)
stream_info = {
    "EQ_ECG_Stream": pylsl.StreamInfo("EQ_ECG_Stream", "ECG", 1, 256, pylsl.cf_string, "Equivital"),
    "EQ_HR_Stream": pylsl.StreamInfo("EQ_HR_Stream", "HR", 1, 0.2, pylsl.cf_string, "Equivital"),
    "EQ_Accel_Stream": pylsl.StreamInfo("EQ_Accel_Stream", "Accel", 1, 256, pylsl.cf_string, "Equivital"),
    "EQ_RR_Stream": pylsl.StreamInfo("EQ_RR_Stream", "RR", 1, 25.6, pylsl.cf_string, "Equivital"),
    "EQ_IR_Stream": pylsl.StreamInfo("EQ_IR_Stream", "IR", 1, 25.6, pylsl.cf_string, "Equivital"),
    "EQ_SkinTemp_Stream": pylsl.StreamInfo("EQ_SkinTemp_Stream", "SkinTemp", 1, 1.0/15, pylsl.cf_string, "Equivital"),
    "EQ_GSR_Stream": pylsl.StreamInfo("EQ_GSR_Stream", "GSR", 1, 16, pylsl.cf_string, "Equivital"),
}

# Create LSL outlets for each stream
outlets = {name: pylsl.StreamOutlet(info) for name, info in stream_info.items()}

print("Simulating LSL replay from file...")

# Open recorded data file
with open("lsl_recorded_data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header

    data_records = list(reader)  # Load all rows into memory for precise timing

# Get initial timestamp
start_time = time.time()
first_sample_time = float(data_records[0][0])

# Stream data at the exact original sampling rate
for row in data_records:
    timestamp, stream_name, sample = float(row[0]), row[1], ast.literal_eval(row[2])  # Convert back to list
    adjusted_time = start_time + (timestamp - first_sample_time)  # Adjusted real-world timestamp

    # Wait until it's time to send this sample
    while time.time() < adjusted_time:
        time.sleep(0.001)

    if stream_name in outlets:
        outlets[stream_name].push_sample([str(sample)])  # Convert to string as LSL channel_format=3
        print(f"Replayed: {stream_name} -> {sample} at {timestamp}")
