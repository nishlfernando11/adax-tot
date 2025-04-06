import time
import pylsl
from pylsl import StreamInlet, resolve_byprop, proc_ALL
import json
import pandas as pd
from datetime import datetime
import re
import socketio
import threading
from collections import deque
import argparse
from dotenv import load_dotenv
import os
import numpy as np

from tot.methods.bfs import solve
from tot.methods.standard import genAdaX
from tot.tasks.adax import AdaXTask

import LSLService as LSL
from app.database.vector_store import VectorStore
from LSL_logger import run_lsl_logger
from StressModel import get_stress_pred
from insert_vectors import process_row
from StressModel import *

load_dotenv(dotenv_path="./.env")

# Load API key from env var (preferred)
SERVER_IP = os.getenv("SERVER_IP")
PORT = os.getenv("PORT")


sio = socketio.Client()
vec = VectorStore()

global task
global xai_agent_type
xai_agent_type = 'NoX' # default

def print_green(msg): print(f"\033[32m{msg}\033[0m")
def print_yellow(msg): print(f"\033[33m{msg}\033[0m")
def print_red(msg): print(f"\033[31m{msg}\033[0m")
def print_cyan(msg): print(f"\033[36m{msg}\033[0m")


@sio.event
def connect():
    print("Connected to server")
    # Start emitting after connection is confirmed
    # i = 0
    # while True:
    #     explanation = f"test reason longer one two three four five nine ten words {i}"
    #     sio.emit("xai_message", {"explanation": explanation})
    #     print("Emitted:", explanation)
    #     i += 1
    #     time.sleep(1)

@sio.on('start_ecg')
def on_start_ecg(data):
    global xai_agent_type
    start_info = data.get('start_info', {})
    print_green("Start ecg data "+ json.dumps(data))
    if start_info["xaiAgentType"] == "AdaX":
        xai_agent_type = 'AdaX'
    elif start_info["xaiAgentType"] == "StaticX":
        xai_agent_type = 'StaticX'
    else:
        xai_agent_type = 'NoX'
    
    print_green(f"{xai_agent_type} agent type detected")
    eeg_inlet = safe_resolve('Emotiv_EEG')
    print("eeg_inlet", eeg_inlet)
    buffer_eeg_data(eeg_inlet)
    
@sio.on('stop_ecg')
def on_end_ecg(data):
    print_green("Ending data collection ")
    # print_green("Ending data collection "+ json.dumps(data))
    

@sio.event
def disconnect():
    print("Disconnected from server")
    
# utility function to parse the behavioral state
def parse_behavioral_state(raw_behavioral_state):
    if isinstance(raw_behavioral_state, list) and len(raw_behavioral_state) > 0:
        try:
            # Step 1: Parse outer JSON string
            outer = json.loads(raw_behavioral_state[0])
            
            # Step 2: Parse the nested \"state\" string
            if "state" in outer:
                outer["state"] = json.loads(outer["state"])
            
            return outer
        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
            return {}
    return {}


def parse_game_state(game_state):
    if isinstance(game_state, list) and len(game_state) > 0:
        try:
            return json.loads(game_state[0])
        except json.JSONDecodeError as e:
            print("Failed to parse game_state:", e)
    return {}

# def map_state_and_features_to_output(game_data, physio_inference, adax_features):
#     parsed_state = json.loads(game_data['state']) #TODO: Use held object and player information in the prompt
#     print("adax_features", adax_features)
#     print_cyan("game_data" + json.dumps(game_data))
#     print_cyan("physio_inference" + json.dumps(physio_inference))
#     # Parsing explanation features from adax_features
#     feature_dict = {feature.split(': ')[0]: feature.split(': ')[1] for feature in adax_features.get('features', [])}

#     physio_data = physio_inference

#     playerId = None
#     if game_data.get("player_0_is_human"):
#         #player 0 is human
#         playerId = game_data.get("player_0_id")
#     else:
#         #player 1 is human
#         playerId = game_data.get("player_1_id")
        
#     output = {
#         "playerId": playerId,
#         "score": game_data.get("score", 0),
#         "trust": physio_data.get("trust", "unknown"),
#         "stress": physio_data.get("stress", "unknown"),
#         "time_left": round(game_data.get("time_left", 0)),
#         "created_at": datetime.utcnow().isoformat(),
#         "cognitive_load": physio_data.get("cognitive_load", "unknown"),
#         "num_collisions": game_data.get("num_collisions", 0),
#         "final_explanation": f"{adax_features.get('answer')}",
#         "justification": f"{adax_features.get('justification')}",
#         "explanation_timing": feature_dict.get("timing", "unknown"),
#         "explanation_duration": feature_dict.get("duration", "unknown"),
#         "explanation_granularity": feature_dict.get("granularity", "unknown")
#     }

#     return output


def map_state_and_features_to_output(game_data, physio_inference, adax_features):
    parsed_state = json.loads(game_data['state'])

    # Parse adaptive explanation features
    feature_dict = adax_features.get('features', {})

    physio_data = physio_inference

    playerId = game_data.get("player_0_id") if game_data.get("player_0_is_human") else game_data.get("player_1_id")

    score = game_data.get("score", 0)
    num_collisions = game_data.get("num_collisions", 0)
    time_elapsed = game_data.get("time_elapsed", 1e-6)
    collision_rate = round(num_collisions / time_elapsed, 2)
    score_rate = round(score / time_elapsed, 2)
    layout_name = game_data.get("layout_name", "unknown")

    output = {
        "playerId": playerId,
        "score": score,
        "trust": physio_data.get("trust", "unknown"),
        "stress": physio_data.get("stress", "unknown"),
        "cognitive_load": physio_data.get("cognitive_load", "unknown"),
        "time_left": round(game_data.get("time_left", 0)),
        "num_collisions": num_collisions,
        "collision_rate": collision_rate,
        "score_rate": score_rate,
        "layout_name": layout_name,
        "final_explanation": adax_features.get("answer", ""),
        "justification": adax_features.get("justification", ""),
        "explanation_timing": feature_dict.get("timing", "unknown"),
        "explanation_duration": feature_dict.get("duration", "unknown"),
        "explanation_granularity": feature_dict.get("granularity", "unknown"),
        "created_at": datetime.utcnow().isoformat()
    }

    return output

def buffer_eeg_data(eeg_inlet=None):

    lsl_start_time = pylsl.local_clock()
    unix_start_time = time.time()
    lsl_to_unix_offset = unix_start_time - lsl_start_time
    eeg_offset = safe_time_correction(eeg_inlet, "Emotiv_EEG")
    # eeg_buffer = []
    # if eeg_inlet: 
    #     eeg_sample, eeg_timestamp = eeg_inlet.pull_sample(timeout=0.0)
    #     if eeg_sample:
    #         print_data(eeg_sample, eeg_timestamp, eeg_offset, "Emotiv_EEG")
    #         channels = eeg_sample['data'][2:34]  # Assuming the first two values and last 4 values are not EEG data
    #         buffer.append(channels)
    global eeg_buffer
    is_data_available = True  # Flag to check if data is available
    while is_data_available:
        # print("Buffer shape:", eeg_buffer.shape)
        # print("Buffer size:", eeg_buffer.size)
        # print("Buffer length:", len(eeg_buffer))   
        eeg_sample, eeg_timestamp = eeg_inlet.pull_sample(timeout=1.0)
        if eeg_sample:
            is_data_available = True  # Data is available
            print_data(eeg_sample, eeg_timestamp, eeg_offset, "Emotiv_EEG")
            # print("eeg_sample", eeg_sample)
            json_sample = json.loads(eeg_sample[0])
            new_sample = json_sample['data'][2:34]  # Assuming the first two values and last 4 values are not EEG data
            new_sample = np.array(new_sample)
            new_sample = new_sample.reshape(32, 1)  # column vector
            eeg_buffer = np.hstack((eeg_buffer, new_sample))  # append along time axis
            
        else:
            is_data_available = False
            print("No data available, waiting...") 
            
            
def get_static_explanation(behavioral_data, physio_inference, timestamp):
    # Todo: get contextual explanation without considering any state information
    args = argparse.Namespace(backend='gpt-3.5-turbo', temperature=0.7, task='adax', xai_agent_type='StaticX', is_local=False, naive_run=False, prompt_sample="standard", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1, return_dataframe=True)
    
    task = AdaXTask(vector_db=vec)

    task.set_data([{"timestamp": timestamp, "behavioral_state": behavioral_data, "physiological_state": physio_inference}])
    ys, infos = solve(args, task, 0, vector_db=vec, parallel=True)
    
    print(infos)
    print(ys)
    # Fix escaped underscore in the JSON string
    fixed_json_str = re.sub(r'\\\\_', '_', ys[0]).replace("True", "true").replace("False", "false").replace("None", "null")
    print(fixed_json_str)
    # Now parse the fixed JSON
    parsed_data = json.loads(fixed_json_str)
    print("parsed_data", parsed_data)
    print("\n\n=========================================\n")
    print(f"Best explanation: {parsed_data['answer']}")
    print(f"Features used: {parsed_data['features']}") 
    print(f"enough_context: {parsed_data['enough_context']}")
    print("\n=========================================\n\n")

    if parsed_data['answer']:
        sio.emit('xai_message', {'explanation': parsed_data['answer'],  'xai_agent_type': xai_agent_type})
    else:
        sio.emit('xai_message', {'explanation': 'I am cooking onion soup. Work with me to cook more dishes.', 'xai_agent_type': xai_agent_type})
    
    #TODO: save entry to RAG
    formatted_data = map_state_and_features_to_output(behavioral_data, physio_inference, parsed_data)
    # print("Formatted data:", formatted_data)
    rag_record = process_row(formatted_data, vec)
    # print("RAG record:", rag_record)
    vec.upsert(rag_record)

    
def get_adax_explanation(behavioral_data, physio_inference, timestamp):
    #TODO: Use held object and player information in the prompt
    # args = argparse.Namespace(backend='mistral:7b-instruct-q4_0', temperature=0.6, task='adax', is_local=True, naive_run=False, prompt_sample="cot" method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=2, n_evaluate_sample=2, n_select_sample=1, stop=None)
    args = argparse.Namespace(backend='gpt-3.5-turbo', temperature=0.7, task='adax', xai_agent_type='AdaX', is_local=False, naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=2, n_evaluate_sample=2, n_select_sample=1, return_dataframe=True)
    
    task = AdaXTask(vector_db=vec)

    task.set_data([{"timestamp": timestamp, "behavioral_state": behavioral_data, "physiological_state": physio_inference}])
    ys, infos = solve(args, task, 0, vector_db=vec, parallel=True)
    
    print(infos)
    print(ys)
    # Fix escaped underscore in the JSON string
    fixed_json_str = re.sub(r'\\\\_', '_', ys[0]).replace("True", "true").replace("False", "false")
    print(fixed_json_str)
    # Now parse the fixed JSON
    parsed_data = json.loads(fixed_json_str)
    print(parsed_data)
    print("\n\n=========================================\n")
    print(f"Best explanation: {parsed_data['answer']}")
    print(f"Features used: {parsed_data['features']}") 
    print(f"enough_context: {parsed_data['enough_context']}")
    print("\n=========================================\n\n")

    sio.emit('xai_message', {'explanation': parsed_data['answer'],  'xai_agent_type': xai_agent_type})
    
    #TODO: save entry to RAG
    formatted_data = map_state_and_features_to_output(behavioral_data, physio_inference, parsed_data)
    # print("Formatted data:", formatted_data)
    rag_record = process_row(formatted_data, vec)
    # print("RAG record:", rag_record)
    vec.upsert(rag_record)
    
# ========== Buffers ==========
ecg_buffer = deque(maxlen=256 * 10)        # 10 seconds of ECG @ 256Hz
game_buffer = deque(maxlen=6 * 10)        # 10 seconds of game data @ 6Hz
# eeg_buffer = deque(maxlen=256 * 10)        # 10 seconds of EEG @ 256Hz
eeg_buffer = np.zeros((32, 0))  # shape: (channels, time_steps)

# ========== Stream Threads ==========
# def stream_ecg():
#     print("Resolving ECG stream...")
#     ecg_stream = resolve_byprop('name', 'EQ_ECG_Stream')  # or ECG.
#     inlet = StreamInlet(ecg_stream[0]) if ecg_stream else None
#     print("ECG stream connected")

#     while True:
#         sample, timestamp = inlet.pull_sample()
#         ecg_buffer.append((timestamp, sample))

def stream_ecg():
    inlet = None

    while True:
        try:
            if inlet is None:
                print("🔄 Resolving ECG stream...")
                streams = resolve_byprop('name', 'EQ_ECG_Stream', timeout=5)
                if not streams:
                    print_red("❌ ECG stream not found. Retrying in 3s...")
                    time.sleep(3)
                    continue
                inlet = StreamInlet(streams[0])
                print_green("✅ ECG stream connected!")

            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                ecg_buffer.append((timestamp, sample))

        except RuntimeError as e:
            print_yellow(f"⚠️ ECG stream lost or disconnected: {e}")
            inlet = None
            time.sleep(3)

        except Exception as e:
            print_red(f"❌ Unexpected error in stream_ecg: {e}")
            inlet = None
            time.sleep(3)


def stream_eeg():
    inlet = None

    while True:
        try:
            if inlet is None:
                print("🔄 Resolving EEG stream...")
                streams = resolve_byprop('name', 'EQ_EEG_Stream', timeout=5)
                if not streams:
                    print_red("❌ EEG stream not found. Retrying in 3s...")
                    time.sleep(3)
                    continue
                inlet = StreamInlet(streams[0])
                print_green("✅ EEG stream connected!")

            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                eeg_buffer.append((timestamp, sample))

        except RuntimeError as e:
            print_yellow(f"⚠️ EEG stream lost or disconnected: {e}")
            inlet = None
            time.sleep(3)

        except Exception as e:
            print_red(f"❌ Unexpected error in stream_eeg: {e}")
            inlet = None
            time.sleep(3)
                        
isGameOn = False
def stream_game():
    global isGameOn
    inlet = None

    while True:
        try:
            if inlet is None:
                print("🔄 Resolving Overcooked stream...")
                streams = resolve_byprop('name', 'OvercookedStream', timeout=5)
                if not streams:
                    print_red("❌ Overcooked stream not found. Retrying in 3s...")
                    time.sleep(3)
                    continue
                inlet = StreamInlet(streams[0])
                print_green("✅ Game stream connected!")

            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                isGameOn = True
                game_buffer.append((timestamp, sample))
            else:
                isGameOn = False

        except RuntimeError as e:
            print_yellow(f"⚠️ Stream lost or disconnected: {e}")
            inlet = None
            isGameOn = False
            time.sleep(3)

        except Exception as e:
            print_red(f"❌ Unexpected error: {e}")
            inlet = None
            isGameOn = False
            time.sleep(3)

# isGameOn = False
# def stream_game():
#     print("Resolving Overcooked stream...")
#     game_stream = resolve_byprop('name', 'OvercookedStream')  # or ECG.
#     inlet = StreamInlet(game_stream[0]) if game_stream else None
#     print("Game stream connected")
#     global isGameOn
#     while True:
#         sample, timestamp = inlet.pull_sample()
#         if sample: 
#             isGameOn = True
#         else:
#             isGameOn = False
#         game_buffer.append((timestamp, sample))

# TODO: Add a function to get the physiometrics from the EEG model
def estimate_cognitive_load(model_output, stress_threshold=0.6, relaxation_threshold=0.4):
    stress = model_output.get("stress", 0)
    relaxation = model_output.get("relaxation", 0)

    if stress >= stress_threshold and relaxation <= relaxation_threshold:
        cognitive_load = "high"
    else:
        cognitive_load = "low"
    
    return cognitive_load

def map_emotions_to_trust_stress(model_output, threshold=0.5):
    #TODO: might want to take the max and handle egde case when all probabilities are the same
    stress_level = "high" if model_output.get("stress", 0) >= threshold else "low" 
    
    positive_emotions = model_output.get("relaxation", 0) + model_output.get("excitement", 0)
    negative_emotions = model_output.get("stress", 0) + model_output.get("depression", 0)

    trust_level = "high" if positive_emotions >= negative_emotions else "low"

    return {
        "stress": stress_level,
        "trust": trust_level,
        "cognitive_load": estimate_cognitive_load(model_output)
    }
    
    
def get_physiometrics(model_output):
    # Placeholder for physiometrics extraction
    # This function should call Stress model and return the result
    '''
    # Example model output 
    #  return {
        "stress": "high",
        "trust": "low",
        "cognitive_load": "high",
    }
    '''

    physio_inference = map_emotions_to_trust_stress(model_output)
    return physio_inference

# ========== Processing Loop ==========
def process_and_explain(physio_window=1.0):
    print("🧠 Starting explanation engine...")
    global isGameOn

    while True:
        if not isGameOn:
            print_yellow("⏸ Game not active. Waiting...")
            time.sleep(0.5)
            continue

        print_green("🎮 Game detected. Beginning live inference...")

        while isGameOn:
            if len(game_buffer) == 0:
                time.sleep(0.05)
                continue
            print_cyan(f"\nGame buffer: {len(game_buffer)}")
            print_cyan(f"\ECG buffer: {len(ecg_buffer)}")
            game_ts, game_state = game_buffer[-1]
            print_cyan(f"\nGame state: {game_state}")

            ecg_window = [s for t, s in ecg_buffer if game_ts - physio_window <= t <= game_ts]
            # eeg_window = [s for t, s in eeg_buffer if game_ts - physio_window <= t <= game_ts]
            print("ECG window:", ecg_window)
            # print("EEG window:", eeg_window)

            # TODO: get real ECG/EEG data and preprocess
            # eeg_data = np.random.randn(32, 768)
            # stress_output = get_stress_pred(eeg_data)
            # print_green("stress_output: "+json.dumps(stress_output))
            # {
                # "prediction": "excitement", 
                # "probabilities": 
                # {"excitement": 0.9462465047836304, "stress": 5.050985419075005e-05, "depression": 0.0007130626472644508, "relaxation": 0.0529899038374424}
            # }
            # TODO: Replace with real model call
            # model_output = {
            #     "excitement": 0.1,
            #     "stress": 0.7,
            #     "depression": 0.1,
            #     "relaxation": 0.1
            # }
            
            # model_output = {
            #     "prediction": "excitement", 
            #     "probabilities": 
            #         {
            #             "excitement": 0.9462465047836304,
            #             "stress": 5.050985419075005e-05,
            #             "depression": 0.0007130626472644508,
            #             "relaxation": 0.0529899038374424
            #             }
            #         }
            model_output = {
                "probabilities" :{
                        "excitement": 0.25,
                        "stress": 0.25,
                        "depression": 0.25,
                        "relaxation": 0.25
                        }}
            global eeg_buffer
            if eeg_buffer.shape[1] == 256*5:
                copy_eeg_buffer = eeg_buffer.copy()
                eeg_buffer = np.zeros((32, 0))  # Reset buffer after processing
                print("Buffer shape after reset:", eeg_buffer.shape)
                print("Buffer shape:", copy_eeg_buffer.shape)
                model_output = get_prediction(copy_eeg_buffer)
                print("\n==============\nPrediction:", model_output)
                
            model_emotion_probs = model_output.get("probabilities")
            physio_inference = get_physiometrics(model_emotion_probs)
            
            behavioral_data = parse_game_state(game_state)
            playerId = None
            if behavioral_data.get("player_0_is_human"):
                #player 0 is human
                playerId = behavioral_data.get("player_0_id")
            else:
                #player 1 is human
                playerId = behavioral_data.get("player_1_id")
            
            behavioral_data["playerId"] = playerId
            print_green(behavioral_data)
            start_time = time.time()
            if xai_agent_type == 'AdaX':
                get_adax_explanation(behavioral_data, physio_inference, game_ts)
                print(f"🧠 Inference time: {time.time() - start_time:.2f}s")
            elif xai_agent_type == 'StaticX':
                get_static_explanation(behavioral_data, physio_inference, game_ts)
                print(f"🧠 Inference time: {time.time() - start_time:.2f}s")
            else:
                print_red("No XAI agent type detected. Skipping explanation.")   
            time.sleep(5)  # ~0.2Hz processing rate


# def process_and_explain(physio_window=1.0):
#     print("Starting explanation loop...")
#     global isGameOn
#     print_green('isGameOn: '+ str(isGameOn))
#     while isGameOn:
#         if len(game_buffer) == 0:
#             time.sleep(0.05)
#             continue

#         # Get most recent game frame
#         game_ts, game_state = game_buffer[-1]
#         print("Game state:", game_state)
#         # Filter recent ECG data within default: 1s window
#         ecg_window = [s for t, s in ecg_buffer if game_ts - physio_window <= t <= game_ts]
#         print("ECG window:", ecg_window)
        
#         #TODO: clean and call Stress model
#         model_output = {
#             "excitement": 0.1,
#             "stress": 0.7,
#             "depression": 0.1,
#             "relaxation": 0.1
#         }
    
#         physio_inference = get_physiometrics(model_output)
#         print("game_state :", game_state)
#         print("ecg_window :", ecg_window)
#         print("physio_inference :", physio_inference)
#         behavioral_data = parse_game_state(game_state)
#         start_time = time.time()
#         # Generate explanation using ToT
#         get_adax_explanation(behavioral_data, physio_inference, game_ts)
#         print(f"------Time taken-- {time.time()-start_time}----------")
#         time.sleep(1 / 6)  # Run at ~6Hz


# ========== Start Everything ==========
if __name__ == '__main__':
    try:
        # connect to the socket server
        sio.connect(f'http://{SERVER_IP}:{PORT}')
        if not sio.connected:
            print_red("Socket connection failed. Retrying in 3s...")
            time.sleep(3)
    
    except Exception as e:
        print_red("Socket Connection failed:", e)
        
    # Start LSL stream listener
    threading.Thread(target=stream_ecg, daemon=True).start()
    threading.Thread(target=stream_eeg, daemon=True).start()
    threading.Thread(target=stream_game, daemon=True).start()
    threading.Thread(target=run_lsl_logger, daemon=True).start()
    threading.Thread(target=process_and_explain, daemon=True).start()

    while True:
        time.sleep(1)  # Keep main thread alive

# Run as a background thread
