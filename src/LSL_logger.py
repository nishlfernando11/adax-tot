from pylsl import StreamInlet, resolve_byprop, proc_ALL
import time
import pylsl
import json
import logging
import csv
import os
from datetime import datetime
import threading

# === Logging Setup ===
log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M') + '.log'
logging.basicConfig(filename=f'datalogs/{log_filename}', 
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# === CSV Setup ===
csv_dir = 'csvlogs'
os.makedirs(csv_dir, exist_ok=True)

stream_names = ['EQ_ECG_Stream', 'EQ_HR_Stream', 'EQ_Accel_Stream', 'EQ_IR_Stream', 'EQ_RR_Stream', 'EQ_SkinTemp_Stream', 'EQ_GSR_Stream', 'Eye_Tracker_Stream', 'OvercookedStream', 'Emotiv_EEG', 'Emotiv_MET']
csv_files = {}
csv_writers = {}
unified_writer = None

player_id = "default"
round_id = "default"
logger_started = False
logger_stop_event = threading.Event()

def setup_csv_files():
    global csv_files, csv_writers, unified_writer

    folder_path = os.path.join(csv_dir, player_id, str(round_id))
    os.makedirs(folder_path, exist_ok=True)

    csv_files = {}
    csv_writers = {}
    for name in stream_names:
        path = f'{folder_path}/{name.lower()}.csv'
        f = open(path, mode='w', newline='')
        writer = csv.DictWriter(f, fieldnames=["timestamp", "data"])
        writer.writeheader()
        csv_files[name] = f
        csv_writers[name] = writer

    unified_path = f'{folder_path}/all_streams.csv'
    unified_csv_file = open(unified_path, mode='w', newline='')
    unified_writer = csv.DictWriter(unified_csv_file, fieldnames=["stream", "timestamp", "data"])
    unified_writer.writeheader()
    csv_files['unified'] = unified_csv_file

def save_to_csv(stream_name, timestamp, data_dict):
    try:
        row = {"timestamp": f"{timestamp:.6f}", "data": json.dumps(data_dict)}
        if stream_name in csv_writers:
            csv_writers[stream_name].writerow(row)
        if unified_writer:
            unified_writer.writerow({"stream": stream_name, "timestamp": f"{timestamp:.6f}", "data": json.dumps(data_dict)})
    except Exception as e:
        logging.error(f"CSV write error for {stream_name}: {e}")

def print_data(data_sample, data_timestamp, data_offset, type):
    sample = str(data_sample[0])
    data_sample = json.loads(sample)
    cor_data_timestamp = data_timestamp + data_offset
    print(f"{type}_sample ", data_sample)
    print(f"---{type} timestamp: {cor_data_timestamp}")
    logging.debug(f"{type} {cor_data_timestamp}, {data_sample}")
    save_to_csv(type, cor_data_timestamp, data_sample)

def safe_resolve(name):
    streams = resolve_byprop('name', name)
    return StreamInlet(streams[0]) if streams else None

def reconnect_stream(name):
    while True:
        try:
            streams = resolve_byprop('name', name)
            if streams:
                logging.info(f"Reconnected to stream: {name}")
                return StreamInlet(streams[0])
            else:
                logging.warning(f"Stream {name} not found, retrying...")
        except Exception as e:
            logging.error(f"Error resolving stream {name}: {e}")
        time.sleep(1)

def pull_sample_safe(inlet, name):
    try:
        return inlet.pull_sample(timeout=0.0)
    except Exception as e:
        logging.warning(f"Stream lost: {name}, error {e} attempting to reconnect...")
        return None, None

def fix_timestamp(timestamp):
    if timestamp is None:
        return None
    return time.time() + (timestamp - pylsl.local_clock())

def safe_time_correction(inlet, name):
    if inlet:
        return inlet.time_correction()
    print(f"Warning: {name} stream not available.")
    return 0.0

def run_lsl_logger():
    global ecg_inlet, hr_inlet, rr_inlet, ir_inlet
    global accel_inlet, skinTemp_inlet, gsr_inlet, eye_inlet, oc_inlet

    # Wait until setup_csv_files() has been called by the socket event
    while not csv_files:
        time.sleep(0.1)

    # Resolve streams
    eye_inlet = safe_resolve('Eye_Tracker_Stream')
    ecg_inlet = safe_resolve('EQ_ECG_Stream')
    hr_inlet = safe_resolve('EQ_HR_Stream')
    rr_inlet = safe_resolve('EQ_RR_Stream')
    ir_inlet = safe_resolve('EQ_IR_Stream')
    skinTemp_inlet = safe_resolve('EQ_SkinTemp_Stream')
    accel_inlet = safe_resolve('EQ_Accel_Stream')
    gsr_inlet = safe_resolve('EQ_GSR_Stream')
    oc_inlet = safe_resolve('OvercookedStream')

    ecg_offset = safe_time_correction(ecg_inlet, "ECG")
    hr_offset = safe_time_correction(hr_inlet, "HR")
    rr_offset = safe_time_correction(rr_inlet, "RR")
    ir_offset = safe_time_correction(ir_inlet, "IR")
    accel_offset = safe_time_correction(accel_inlet, "Accel")
    skinTemp_offset = safe_time_correction(skinTemp_inlet, "SkinTemp")
    gsr_offset = safe_time_correction(gsr_inlet, "GSR")
    eye_offset = safe_time_correction(eye_inlet, "Eye Tracker")
    oc_offset = safe_time_correction(oc_inlet, "Overcooked")

    try:
        while not logger_stop_event.is_set():
            for name, inlet_ref, offset in [
                ("ECG", ecg_inlet, ecg_offset),
                ("HR", hr_inlet, hr_offset),
                ("Accel", accel_inlet, accel_offset),
                ("IR", ir_inlet, ir_offset),
                ("RR", rr_inlet, rr_offset),
                ("SkinTemp", skinTemp_inlet, skinTemp_offset),
                ("GSR", gsr_inlet, gsr_offset),
            ]:
                sample, timestamp = pull_sample_safe(inlet_ref, name)
                if sample is None:
                    locals()[f"{name.lower()}_inlet"] = reconnect_stream(f"EQ_{name}_Stream") if name != "Eye" and name != "Overcooked" else reconnect_stream(name + ("_Stream" if name != "Eye" else "_Tracker_Stream"))
                    continue
                sample_data = json.loads(str(sample[0]))
                cor_timestamp = timestamp + offset
                logging.debug(f"{name} {cor_timestamp}, {sample_data}")
                save_to_csv(name, cor_timestamp, sample_data)

            eye_sample, eye_timestamp = pull_sample_safe(eye_inlet, "Eye")
            if eye_sample:
                print_data(eye_sample, eye_timestamp, eye_offset, "Eye")
            elif eye_sample is None:
                eye_inlet = reconnect_stream("Eye_Tracker_Stream")

            oc_sample, oc_timestamp = pull_sample_safe(oc_inlet, "Overcooked")
            if oc_sample:
                print_data(oc_sample, oc_timestamp, oc_offset, "Overcooked")
            elif oc_sample is None:
                oc_inlet = reconnect_stream("OvercookedStream")

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        for f in csv_files.values():
            f.close()
        if 'unified' in csv_files:
            csv_files['unified'].close()
        print("Files closed.")
        logger_stop_event.clear()
        globals()['logger_started'] = False

# This function can be triggered by your socket event
def on_start_ecg(data):
    global player_id, round_id, logger_started
    print("Start ecg data", json.dumps(data))
    player_id = data.get("start_info", {}).get("player_id", "unknown")
    round_id = data.get("start_info", {}).get("round_id", "unknown")
    setup_csv_files()

    if not logger_started:
        logger_started = True
        threading.Thread(target=run_lsl_logger, daemon=True).start()

# This function can be triggered by a socket event to stop logging
def on_stop_ecg():
    global logger_stop_event
    print("Stopping ECG logging...")
    logger_stop_event.set()
