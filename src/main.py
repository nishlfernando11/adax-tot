import time
import pylsl
from pylsl import StreamInlet, resolve_byprop
import json
import csv
import os
import logging
import re
import sys

import pandas as pd
from datetime import datetime
import socketio
import threading
from collections import deque
import argparse
from dotenv import load_dotenv
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
DEBUG_LOGGEER = os.getenv("DEBUG_LOGGER")
METRIC_LOGGEER = os.getenv("METRIC_LOGGEER")


sio = socketio.Client()
vec = VectorStore()

global task
global xai_agent_type
xai_agent_type = 'NoX' # default

def print_green(msg): print(f"\033[32m{msg}\033[0m")
def print_yellow(msg): print(f"\033[33m{msg}\033[0m")
def print_red(msg): print(f"\033[31m{msg}\033[0m")
def print_cyan(msg): print(f"\033[36m{msg}\033[0m")


    
# ========== Buffers ==========
global eeg_buffer, ecg_buffer

ecg_buffer = np.zeros((2, 0))  # shape: (channels, time_steps)
game_buffer = deque(maxlen=6 * 10)        # 10 seconds of game data @ 6Hz
# ecg_buffer = deque(maxlen=256 * 10)        # 10 seconds of game data @ 6Hz
# eeg_buffer = deque(maxlen=256 * 10)        # 10 seconds of EEG @ 256Hz
eeg_buffer = np.zeros((32, 0))  # shape: (channels, time_steps)
# met_buffer = np.zeros((13, 0))  # shape: (features, time_steps)

ecg_buffer_lock = threading.Lock()
eeg_buffer_lock = threading.Lock()
met_buffer_lock = threading.Lock()

met_buffer = []
met_buffer_lock = threading.Lock()

met_labels = ['eng.isActive', 'eng', 'exc.isActive', 'exc', 'lex', 'str.isActive', 'str', 'rel.isActive', 'rel', 'int.isActive', 'int', 'foc.isActive', 'foc']

socket_connected = threading.Event()


# Setup directory and CSV path
csv_dir = METRIC_LOGGEER or 'metrics_logs'
os.makedirs(csv_dir, exist_ok=True)
csv_path = os.path.join(csv_dir, f"metrics_log_{datetime.now().strftime('%Y-%m-%d_%H')}.csv")

# Define the CSV headers
CSV_FIELDS = ["timestamp", "player_id", "round_id", "uid", "type", "data"]

# Write header if file doesn't exist
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

# === Function to write a sample_obj ===
def write_sample(sample_obj):
    row = {
        "timestamp": sample_obj["timestamp"],
        "player_id": sample_obj["player_id"],
        "round_id": sample_obj["round_id"],
        "uid": sample_obj["uid"],
        "type": sample_obj["type"],
        "data": json.dumps(sample_obj["data"])  # <-- this keeps the dict as a single string
    }
    
    # timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_path = os.path.join(csv_dir, f"{sample_obj['uid']}_{sample_obj['player_id']}", str(sample_obj["round_id"]))
    os.makedirs(folder_path, exist_ok=True)

    filename = f"uid{sample_obj['uid']}_round{sample_obj['round_id']}.csv"
    csv_path = os.path.join(folder_path, filename)

    # Write header if file doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
    

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)

# Background thread to retry socket connection
# def retry_socket_connection():
#     while not socket_connected.is_set():
#         try:
#             sio.connect(f'http://{SERVER_IP}:{PORT}')
#             socket_connected.set()
#             print_green("Socket connection established.")
#         except Exception as e:
#             print(f"Socket connection failed, retrying: {e}")
#             time.sleep(5)
         
def retry_socket_connection():
    while not socket_connected.is_set():
        try:
            if not sio.connected:
                print("🔄 Attempting socket connection...")
                sio.connect(f'http://{SERVER_IP}:{PORT}', namespaces=["/"])
                socket_connected.set()
                print_green("✅ Socket connection established.")
            # else:
            #     print_yellow("⚠️ Socket already connected or connecting.")
        except Exception as e:
            print_red(f"Socket connection failed, retrying: {e}")
            time.sleep(5)


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

xai_agent_type = 'NoX' # default
current_uid = None
current_round_id = None

@sio.on('start_sensors')
def on_start_sensors(data):
    global xai_agent_type, current_uid, current_round_id
    start_info = data.get('start_info', {})
    print_green("Start ecg data "+ json.dumps(data))
    if start_info["xaiAgentType"] == "AdaX":
        xai_agent_type = 'AdaX'
    elif start_info["xaiAgentType"] == "StaticX":
        xai_agent_type = 'StaticX'
    else:
        xai_agent_type = 'NoX'
        
    current_uid = start_info.get("uid", None) 
    current_round_id = start_info.get("round_id", None)  
    
    print_green(f"{xai_agent_type} agent type detected")
    # eeg_inlet = safe_resolve('Emotiv_EEG')
    # print("eeg_inlet", eeg_inlet)
    # buffer_eeg_data(eeg_inlet)
    
@sio.on('stop_sensors')
def on_end_sensors(data):
    print_green("Ending data collection ")
    global current_uid
    current_uid = None # reset
    # print_green("Ending data collection "+ json.dumps(data))
    

@sio.event
def disconnect():
    print("Disconnected from server")
    socket_connected.clear()
    threading.Thread(target=retry_socket_connection).start()
    
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
    import json
    from datetime import datetime

    # Parse game state
    parsed_state = json.loads(game_data.get("state", "{}"))
    players = parsed_state.get("players", [])
    joint_action = json.loads(game_data.get("joint_action", "[[0,0],[0,0]]"))

    # Determine which player is human and AI
    player_roles = {}
    if game_data.get("player_0_is_human", True):
        player_roles = {
            "player_0": "(human)player_0",
            "player_1": "(AI)player_1"
        }
    else:
        player_roles = {
            "player_0": "(AI)player_0",
            "player_1": "(human)player_1"
        }

    # Decode joint actions to simple text
    action_map = {
        (0, 0): "idle",
        (0, 1): "move down",
        (0, -1): "move up",
        (1, 0): "move right",
        (-1, 0): "move left",
        5: "interact",
        6: "pickup",
        7: "drop"
    }
    
    # Normalize joint_action values (they might be ints like 5, 6, 7)
    def decode_action(act):
        act_tuple = tuple(act) if isinstance(act, list) else act
        return action_map.get(act_tuple, "unknown")

    player_actions = {
        player_roles["player_0"]: decode_action(joint_action[0]),
        player_roles["player_1"]: decode_action(joint_action[1])
    }
    # player_actions = {
    #     player_roles["player_0"]: action_map.get(tuple(joint_action[0]), "unknown"),
    #     player_roles["player_1"]: action_map.get(tuple(joint_action[1]), "unknown")
    # }

    # Held objects
    held_objects = {
        player_roles["player_0"]: players[0].get("held_object", "nothing") if players[0].get("held_object") else "nothing",
        player_roles["player_1"]: players[1].get("held_object", "nothing") if players[1].get("held_object") else "nothing"
    }

    # Pot status
    pots = [obj for obj in parsed_state.get("objects", []) if obj.get("name") == "pot"]
    pot_status = []
    for pot in pots:
        onions = pot.get("num_items", 0)
        cooking = pot.get("is_cooking", False)
        status = {
            "position": pot.get("position", []),
            "onions": onions,
            "is_cooking": cooking
        }
        pot_status.append(status)

    # AI summary
    ai_key = next(k for k in player_roles.values() if "AI" in k)
    # ai_action_summary = f"{ai_key} {player_actions[ai_key]}, holding {held_objects[ai_key]}"
    ai_summary = f"{ai_key} {player_actions[ai_key]}, holding {held_objects[ai_key]}, position {players[1]['position']}"

    # Adaptive features
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
        "timestep": parsed_state.get("timestep", 0),
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
        "created_at": datetime.utcnow().isoformat(),

        # Agent-tagged additions
        "player_actions": player_actions,
        "held_objects": held_objects,
        "ai_action_summary": ai_summary,
        "pot_status": pot_status
    }

    return output


def map_state_and_features_to_output_old(game_data, physio_inference, adax_features):
    print_cyan("game_data" + json.dumps(game_data))
    print_cyan("physio_inference" + json.dumps(physio_inference
                                               ))
    print_cyan("adax_features" + json.dumps(adax_features))
    # parsed_state = json.loads(game_data['state'])

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

    # TODO: ADD UID, round_id, etc. to the output
    output = {
        "timestep": game_data.get("timestep", 0),
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

    # lsl_start_time = pylsl.local_clock()
    # unix_start_time = time.time()
    # lsl_to_unix_offset = unix_start_time - lsl_start_time
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
            time.sleep(0.5)  # let CPU rest briefly
            
            
def get_static_explanation(behavioral_data, physio_inference, timestamp, model_output):
    # Todo: get contextual explanation without considering any state information
    # args = argparse.Namespace(backend='gpt-3.5-turbo', temperature=0.6, task='adax', xai_agent_type='StaticX', is_local=False, naive_run=False, prompt_sample="standard", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1, return_dataframe=True)
    
    task = AdaXTask(vector_db=vec)

    task.set_data([{"timestamp": timestamp, "behavioral_state": behavioral_data, "physiological_state": physio_inference}])
    # ys, infos = solve(args, task, 0, vector_db=vec, parallel=True)
    
    # print(infos)
    # print(ys)
    # # Fix escaped underscore in the JSON string
    # fixed_json_str = re.sub(r'\\\\_', '_', ys[0]).replace("True", "true").replace("False", "false").replace("None", "null")
    # print(fixed_json_str)
    # # Now parse the fixed JSON
    # parsed_data = json.loads(fixed_json_str)
    # print("parsed_data", parsed_data)
    # print("\n\n=========================================\n")
    # print(f"Best explanation: {parsed_data['answer']}")
    # print(f"Features used: {parsed_data['features']}") 
    # print(f"enough_context: {parsed_data['enough_context']}")
    # print("\n=========================================\n\n")
    
    static_explanation = task.standard_rule_based_explanation(behavioral_data)

    if static_explanation:
        sio.emit('xai_message', {'explanation': static_explanation,  'xai_agent_type': xai_agent_type})
    else:
        sio.emit('xai_message', {'explanation': 'I am cooking onion soup.', 'xai_agent_type': xai_agent_type})

    formatted_data = map_state_and_features_to_output(behavioral_data, physio_inference, {})
    # print("Formatted data:", formatted_data)
    rag_record = process_row(formatted_data, vec)
    # print("RAG record:", rag_record)
    vec.upsert(rag_record)
    
    save_metrics_data(behavioral_data, behavioral_data.get("playerId"), model_output)
    
def get_adax_explanation(behavioral_data, physio_inference, timestamp, model_output):
    #TODO: Use held object and player information in the prompt
    # args = argparse.Namespace(backend='mistral:7b-instruct-q4_0', temperature=0.6, task='adax', xai_agent_type='AdaX', verify_with_gpt=True, is_local=True, naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=2, n_evaluate_sample=2, n_select_sample=1, return_dataframe=True)
    args = argparse.Namespace(backend='gpt-4.1-nano', temperature=0.6, task='adax', xai_agent_type='AdaX', verify_with_gpt=True, is_local=False, naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=2, n_evaluate_sample=2, n_select_sample=1, return_dataframe=True)
    #gpt-4.1-mini
    #gpt-3.5-turbo
    # gpt-4o-mini
    #gpt-4o
    ## not very good
    # gpt-4.1-nano
    #gpt-4o-realtime-preview
    task = AdaXTask(vector_db=vec)
    
    task.set_data([{"timestamp": timestamp, "behavioral_state": behavioral_data, "physiological_state": physio_inference}])

    print_green("Starting explanation generation...")
    start = time.time()
    ys, infos = solve(args, task, 0, vector_db=vec, parallel=True, xai_agent_type='AdaX')
    print("⏱ solve() time:", round(time.time() - start, 2), "s")
    
    static_explanation = task.standard_rule_based_explanation(behavioral_data)

    if static_explanation:
        sio.emit('xai_message', {'explanation': static_explanation,  'xai_agent_type': xai_agent_type})
    else:
        sio.emit('xai_message', {'explanation': 'I am cooking onion soup.', 'xai_agent_type': xai_agent_type})

    formatted_data = map_state_and_features_to_output(behavioral_data, physio_inference, {})
    # print("Formatted data:", formatted_data)
    rag_record = process_row(formatted_data, vec)
    # print("RAG record:", rag_record)
    vec.upsert(rag_record)
    
    save_metrics_data(behavioral_data, behavioral_data.get("playerId"), model_output)
    
def get_adax_explanation_old(behavioral_data, physio_inference, timestamp, model_output):
    #TODO: Use held object and player information in the prompt
    # args = argparse.Namespace(backend='mistral:7b-instruct-q4_0', temperature=0.6, task='adax', xai_agent_type='AdaX', verify_with_gpt=True, is_local=True, naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=2, n_evaluate_sample=2, n_select_sample=1, return_dataframe=True)
    args = argparse.Namespace(backend='gpt-4.1-mini', temperature=0.6, task='adax', xai_agent_type='AdaX', verify_with_gpt=True, is_local=False, naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=2, n_evaluate_sample=2, n_select_sample=1, return_dataframe=True)
    #gpt-4.1-mini
    #gpt-3.5-turbo
    task = AdaXTask(vector_db=vec)
    
    task.set_data([{"timestamp": timestamp, "behavioral_state": behavioral_data, "physiological_state": physio_inference}])

    # Optional: during the delay send a static explanation
    static_explanation = task.standard_rule_based_explanation(behavioral_data)
    if static_explanation:
        sio.emit('xai_message', {
            'explanation': static_explanation,
            'xai_agent_type': xai_agent_type,
            'fallback': True  # optional flag to show it's static
        })
    print_green("Starting explanation generation...")
    start = time.time()
    ys, infos = solve(args, task, 0, vector_db=vec, parallel=True)
    print("⏱ solve() time:", round(time.time() - start, 2), "s")
    
    # print(infos)
    # print(ys)
    # Fix escaped underscore in the JSON string
    fixed_json_str = re.sub(r'\\\\_', '_', ys[0]).replace("True", "true").replace("False", "false")
    # print(fixed_json_str)
    # Now parse the fixed JSON
    parsed_data = json.loads(fixed_json_str)
    # print(parsed_data)
    print_green("\n\n=========================================\n")
    print_green(f"Best explanation: {parsed_data['answer']}")
    print_green(f"Features used: {parsed_data['features']}") 
    print_green(f"enough_context: {parsed_data['enough_context']}")
    print_green("\n=========================================\n\n")

    sio.emit('xai_message', {'explanation': str(parsed_data['answer']),  'xai_agent_type': xai_agent_type})
    
    #TODO: save entry to RAG
    formatted_data = map_state_and_features_to_output(behavioral_data, physio_inference, parsed_data)
    # print("Formatted data:", formatted_data)
    rag_record = process_row(formatted_data, vec)
    # print("RAG record:", rag_record)
    vec.upsert(rag_record)
    save_metrics_data(behavioral_data, behavioral_data.get("playerId"), model_output)

def save_metrics_data(behavioral_data, playerId, model_output): 
    print_red("met_buffer ")
    print(met_buffer)
    print("behavioral_data")
    print_cyan(behavioral_data)
    if len(met_buffer) == 0:
        print_red("No metrics data to save")
    #save met data and stress model data into csv
    with met_buffer_lock:
        for entry in met_buffer:
            if isinstance(entry, tuple) and len(entry) == 2:
                timestamp, sample = entry
                sample_obj = {
                    "timestamp": timestamp,
                    "player_id": playerId,
                    "round_id": current_round_id,
                    "uid": current_uid,
                    "type": "met",
                    "data": dict(zip(met_labels, sample))
                }
                write_sample(sample_obj)
            else:
                print(f"Invalid entry in met_buffer: {entry}")
        # clear buffer
        met_buffer.clear()
    # for (timestamp, sample) in met_buffer:
    #     sample_obj = {
    #         "timestamp": timestamp,
    #         "player_id": playerId,
    #         "round_id": behavioral_data.get("round_id"),
    #         "uid": current_uid,
    #         "type": "met",
    #         "data": dict(zip(met_labels, sample))
    #     }
    #     write_sample(sample_obj)
    # Write model prediction
    sample_obj = {
        "timestamp": model_output.get("timestamp"),
        "player_id": playerId,
        "round_id": current_round_id,
        "uid": current_uid,
        "type": "model",
        "data": model_output
    }
    write_sample(sample_obj)
    
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
    global ecg_buffer

    while True:
        try:
            if inlet is None:
                print("🔄 Resolving ECG stream...")
                streams = resolve_byprop('name', 'EQ_ECG_Stream', timeout=5)
                if not streams:
                    print_red("❌ ECG stream not found. Retrying in 3s...")
                    time.sleep(3)
                    continue
                print_green("ECG stream found!")
                print(streams[0])
                inlet = StreamInlet(streams[0])
                print_green("✅ ECG stream connected!")
                print(inlet.info())

            # sample, timestamp = inlet.pull_sample(timeout=1.0)
            # if sample:
            #     ecg_buffer.append((timestamp, sample))
                
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            # print("📥 ECG raw sample:", sample)
            if sample:
                try:
                    if isinstance(sample[0], dict):
                        json_sample = sample[0]
                    elif isinstance(sample[0], str):
                        json_sample = json.loads(sample[0])
                    else:
                        print_red(f"⚠️ Unsupported sample format: {type(sample[0])}, sample: {sample}")
                        continue
                except Exception as e:
                    print_red("❌ Failed to parse ECG sample")
                    import traceback; traceback.print_exc()
                    continue
                new_sample = np.array([json_sample['lead_one_mv'], json_sample['lead_two_mv']]).reshape(2, 1)
                ecg_buffer = np.hstack((ecg_buffer, new_sample))
                with ecg_buffer_lock:
                    ecg_buffer = np.hstack((ecg_buffer, new_sample))
                    # print_cyan(f"✅ ECG buffer updated: {ecg_buffer.shape}")
                
            # ecg_sample, timestamp = inlet.pull_sample(timeout=1.0)
            # if ecg_sample:
            #     print("📥 Raw ECG Sample:", ecg_sample)
            #     try:
            #         # Check format
            #         if isinstance(ecg_sample[0], dict):
            #             json_sample = ecg_sample[0]
            #         elif isinstance(ecg_sample[0], str):
            #             json_sample = json.loads(ecg_sample[0])
            #         else:
            #             raise ValueError("Invalid ECG sample format")

            #         new_sample = np.array([json_sample['lead_one_mv'], json_sample['lead_two_mv']]).reshape(2, 1)

            #         with ecg_buffer_lock:
            #             ecg_buffer = np.hstack((ecg_buffer, new_sample))
            #             print_cyan(f"✅ ECG buffer updated: {ecg_buffer.shape}")

            #     except Exception as e:
            #         print_red(f"❌ Failed to parse ECG sample: {e}")
            #         import traceback; traceback.print_exc()

        except RuntimeError as e:
            print_yellow(f"⚠️ ECG stream lost: {e}")
            inlet = None
            time.sleep(3)
        except Exception as e:
            print_red(f"❌ Unexpected ECG error: {e}")
            import traceback; traceback.print_exc()
            inlet = None
            time.sleep(3)


def stream_ecg_old():
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
            print(inlet.info())

            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                try:
                    if isinstance(sample[0], dict):
                        json_sample = sample[0]
                    elif isinstance(sample[0], str):
                        json_sample = json.loads(sample[0])
                    else:
                        print_red(f"⚠️ Unsupported sample format: {type(sample[0])}, sample: {sample}")
                        continue
                except Exception as e:
                    print_red("❌ Failed to parse ECG sample")
                    import traceback; traceback.print_exc()
                    continue
                new_sample = np.array([json_sample['lead_one_mv'], json_sample['lead_two_mv']]).reshape(2, 1)
                # ecg_buffer = np.hstack((ecg_buffer, new_sample))
                with ecg_buffer_lock:
                    ecg_buffer = np.hstack((ecg_buffer, new_sample))
                    # print_cyan(f"✅ ECG buffer updated: {ecg_buffer.shape}")
                
            # ecg_sample, timestamp = inlet.pull_sample(timeout=1.0)
            # if ecg_sample:
            #     print("📥 Raw ECG Sample:", ecg_sample)
            #     try:
            #         # Check format
            #         if isinstance(ecg_sample[0], dict):
            #             json_sample = ecg_sample[0]
            #         elif isinstance(ecg_sample[0], str):
            #             json_sample = json.loads(ecg_sample[0])
            #         else:
            #             raise ValueError("Invalid ECG sample format")

            #         new_sample = np.array([json_sample['lead_one_mv'], json_sample['lead_two_mv']]).reshape(2, 1)

            #         with ecg_buffer_lock:
            #             ecg_buffer = np.hstack((ecg_buffer, new_sample))
            #             print_cyan(f"✅ ECG buffer updated: {ecg_buffer.shape}")

            #     except Exception as e:
            #         print_red(f"❌ Failed to parse ECG sample: {e}")
            #         import traceback; traceback.print_exc()

        except RuntimeError as e:
            print_yellow(f"⚠️ ECG stream lost: {e}")
            inlet = None
            time.sleep(3)
        except Exception as e:
            print_red(f"❌ Unexpected ECG error: {e}")
            import traceback; traceback.print_exc()
            inlet = None
            time.sleep(3)


def stream_ecg_old():
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
            print(inlet.info())

            # Instead of blocking call:
            ecg_sample, ecg_timestamp = inlet.pull_sample(timeout=0.1)
            print("Sample[0] type:", type(ecg_sample[0]))

            if ecg_sample:
                print("Raw ECG Sample:", ecg_sample)
                # handle sample
                ecg_offset = safe_time_correction(inlet, "EQ_ECG_Stream")
                print_data(ecg_sample, ecg_timestamp, ecg_offset, "EQ_ECG_Stream")
                
                # json_sample = json.loads(ecg_sample[0])
                json_sample = json.loads(ecg_sample[0])
                # "{""round_id"": ""13"", ""player_id"": ""14f8384d4ac144ddb9471b42b08d2a58"", ""uid"": ""3"", ""lead_one_raw"": 0, ""lead_two_raw"": 0, ""sequence_number"": 0, ""lead_one_mv"": -5.194752, ""lead_two_mv"": -5.194752, ""event_time"": 1744686659.1040804, ""lsl_timestamp"": 5815.8237514, ""unix_timestamp"": 1744686659.165712}"

                new_sample = np.array([json_sample['lead_one_mv'],json_sample['lead_two_mv']]).reshape(2, 1)
                print("New ECG Sample:", new_sample)
                print("New ECG Sample shape:", new_sample.shape)
                # with ecg_buffer_lock:
                global ecg_buffer
                ecg_buffer = np.hstack((ecg_buffer, new_sample))

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
                streams = resolve_byprop('name', 'Emotiv_EEG', timeout=5)
                if not streams:
                    print_red("❌ EEG stream not found. Retrying in 3s...")
                    time.sleep(3)
                    continue
                inlet = StreamInlet(streams[0])
                print_green("✅ EEG stream connected!")
            # Instead of blocking call:
            eeg_sample, timestamp = inlet.pull_sample(timeout=1.0)

            if eeg_sample:
                try:
                    if isinstance(eeg_sample[0], dict):
                        json_sample = eeg_sample[0]
                    elif isinstance(eeg_sample[0], str):
                        json_sample = json.loads(eeg_sample[0])
                    else:
                        print_red(f"⚠️ Unsupported sample format: {type(eeg_sample[0])}, sample: {eeg_sample}")
                        continue
                except Exception as e:
                    print_red("❌ Failed to parse ECG sample")
                    import traceback; traceback.print_exc()
                    continue
                # handle sample
                # eeg_offset = safe_time_correction(inlet, "Emotiv_EEG")
                # print_data(eeg_sample, timestamp, eeg_offset, "Emotiv_EEG")
                
                # json_sample = json.loads(eeg_sample[0])
                new_sample = np.array(json_sample['data'][2:34]).reshape(32, 1)

                with eeg_buffer_lock:
                    global eeg_buffer
                    eeg_buffer = np.hstack((eeg_buffer, new_sample))

            else:
                time.sleep(0.01)  # Rest CPU briefly if no sample

        except RuntimeError as e:
            print_yellow(f"⚠️ EEG stream lost or disconnected: {e}")
            inlet = None
            time.sleep(3)

        except Exception as e:
            print_red(f"❌ Unexpected error in stream_eeg: {e}")
            inlet = None
            time.sleep(3)


def stream_metrics():
    inlet = None

    while True:
        try:
            if inlet is None:
                print("🔄 Resolving Metrics stream...")
                streams = resolve_byprop('name', 'Emotiv_MET', timeout=5)
                if not streams:
                    print_red("❌ Metrics stream not found. Retrying in 3s...")
                    time.sleep(3)
                    continue
                inlet = StreamInlet(streams[0])
                print_green("✅ Metrics stream connected!")

            # Instead of blocking call:
            met_sample, met_timestamp = inlet.pull_sample(timeout=0.0)


            if met_sample:
                try:
                    if isinstance(met_sample[0], dict):
                        json_sample = met_sample[0]
                    elif isinstance(met_sample[0], str):
                        json_sample = json.loads(met_sample[0])
                    else:
                        print_red(f"⚠️ Unsupported sample format: {type(met_sample[0])}, sample: {met_sample}")
                        continue
                except Exception as e:
                    print_red("❌ Failed to parse ECG sample")
                    import traceback; traceback.print_exc()
                    continue
                try:
                    # met_offset = safe_time_correction(inlet, "Emotiv_MET")
                    # print_data(met_sample, met_timestamp, met_offset, "Emotiv_MET")
                
                    # json_sample = json.loads(met_sample[0])
                    new_sample = np.array(json_sample['data']).reshape(13, 1)
                    timestamp = json_sample['timestamp']
                    with met_buffer_lock:
                        met_buffer.append((timestamp, new_sample.flatten().tolist()))
                except Exception as e:
                    print(f"❌ Error processing met_sample: {e}")
                    
            # if met_sample:
            #     # handle sample
            #     met_offset = safe_time_correction(inlet, "Emotiv_MET")
            #     print_data(met_sample, met_timestamp, met_offset, "Emotiv_MET")
                
            #     json_sample = json.loads(met_sample[0])
            #     new_sample = np.array(json_sample['data']).reshape(13, 1)
            #     timestamp = json_sample['timestamp']
            #     with met_buffer_lock:
            #         global met_buffer
            #         met_buffer = np.hstack((met_buffer, (timestamp, new_sample.flatten().tolist()))

            else:
                time.sleep(0.01)  # Rest CPU briefly if no sample

        except RuntimeError as e:
            print_yellow(f"⚠️ Metrics stream lost or disconnected: {e}")
            inlet = None
            time.sleep(3)

        except Exception as e:
            print_red(f"❌ Unexpected error in stream_met: {e}")
            inlet = None
            time.sleep(3)


# def stream_eeg():
#     inlet = None

#     while True:
#         try:
#             if inlet is None:
#                 print("🔄 Resolving EEG stream...")
#                 streams = resolve_byprop('name', 'Emotiv_EEG', timeout=5)
#                 if not streams:
#                     print_red("❌ EEG stream not found. Retrying in 3s...")
#                     time.sleep(3)
#                     continue
#                 inlet = StreamInlet(streams[0])
#                 print_green("✅ EEG stream connected!")

#             buffer_eeg_data(inlet)
#             # sample, timestamp = inlet.pull_sample(timeout=1.0)
#             # if sample:
#             #     eeg_buffer.append((timestamp, sample))

#         except RuntimeError as e:
#             print_yellow(f"⚠️ EEG stream lost or disconnected: {e}")
#             inlet = None
#             time.sleep(3)

#         except Exception as e:
#             print_red(f"❌ Unexpected error in stream_eeg: {e}")
#             inlet = None
#             time.sleep(3)
                        
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

# REQUIRED_SIZE = 256  # 256Hz * 1 seconds

# # Trim or pad the data
# def fix_buffer_size(eeg_buffer, target_size=REQUIRED_SIZE):
#     print("fix_buffer_size")
#     print("eeg_buffer shape:", eeg_buffer.shape)
#     current_size = eeg_buffer.shape[1]
#     if current_size > target_size:
#         eeg_buffer = eeg_buffer[:, :target_size]  # truncate
#     elif current_size < target_size:
#         pad_width = target_size - current_size
#         eeg_buffer = np.pad(eeg_buffer, ((0,0), (0,pad_width)), mode='constant')
#     print("new eeg_buffer shape:", eeg_buffer.shape)
#     return eeg_buffer


# ========== Processing Loop ==========
def process_and_explain(physio_window=5.0):
    print("🧠 Starting explanation engine...")
    global isGameOn, ecg_buffer, eeg_buffer

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
            game_ts, game_state = game_buffer[-1] #extract the last game state
            print_cyan(f"\nGame state: {game_state}")

            # ecg_window = [s for t, s in ecg_buffer if game_ts - physio_window <= t <= game_ts]
            # eeg_window = [s for t, s in eeg_buffer if game_ts - physio_window <= t <= game_ts]
            # print("ECG window:", {len(ecg_window)})
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
                "timestamp": time.time(),
                "probabilities" :{
                        "excitement": 0.25,
                        "stress": 0.25,
                        "depression": 0.25,
                        "relaxation": 0.25
                        }}
            print_cyan(f"\EEG buffer shape: {eeg_buffer.shape[1]}")
            print_cyan(f"\ECG buffer shape: {ecg_buffer.shape[1]}")
            
            if eeg_buffer.shape[1] >= 256 and ecg_buffer.shape[1] >= 256: 
                # predict stress level from ecg and eeg
                # with eeg_buffer_lock and ecg_buffer_lock:
                print_cyan(f"\nEEG buffer shape: {eeg_buffer.shape[1]}")
                # print_cyan(f"\nECG buffer shape: {ecg_buffer.shape[1]}")
                # Before sending data
                # if eeg_buffer.shape[1] >= 256 and ecg_buffer.shape[1] >= 256:
                # copy_eeg_buffer = fix_buffer_size(eeg_buffer.copy())
                # copy_ecg_buffer = fix_buffer_size(ecg_buffer.copy())
                copy_eeg_buffer = eeg_buffer.copy()
                copy_ecg_buffer = ecg_buffer.copy()
                eeg_buffer = np.zeros((32, 0))  # Reset buffer after processing
                ecg_buffer = np.zeros((2, 0))  # Reset buffer after processing
                print(f"Buffer shape after reset: EEG: {eeg_buffer.shape}, ECG: {ecg_buffer.shape}")
                print(f"Copy Buffer shape after reset: EEG: {copy_eeg_buffer.shape}, ECG: {copy_ecg_buffer.shape}")
                model_output = get_prediction(eeg_data=copy_eeg_buffer, ecg_data=copy_ecg_buffer)
                print("\n==============\nPrediction:", model_output)
            if eeg_buffer.shape[1] >= 256: 
                # with eeg_buffer_lock:
                # if eeg_buffer.shape[1] >= 256 * 5:
                #     copy_eeg_buffer = eeg_buffer.copy()
                #     eeg_buffer = np.zeros((32, 0))

                print_cyan(f"\nEEG buffer shape: {eeg_buffer.shape[1]}")
                # Before sending data
                # if eeg_buffer.shape[1] >= 256:
                # copy_eeg_buffer = fix_buffer_size(eeg_buffer.copy())
                copy_eeg_buffer = eeg_buffer.copy()
                eeg_buffer = np.zeros((32, 0))  # Reset buffer after processing
                print("Buffer shape after reset:", eeg_buffer.shape)
                print("Buffer shape:", copy_eeg_buffer.shape)
                model_output = get_prediction(eeg_data=copy_eeg_buffer)
                print("\n==============\nPrediction:", model_output)
            if ecg_buffer.shape[1] >= 256: 
                # with ecg_buffer_lock:
                print_cyan(f"\nECG buffer shape: {ecg_buffer.shape[1]}")
                # Before sending data
                # if ecg_buffer.shape[1] >= 256:
                # copy_ecg_buffer" = fix_buffer_size(ecg_buffer.copy())
                copy_ecg_buffer = ecg_buffer.copy()
                ecg_buffer = np.zeros((2, 0))  # Reset buffer after processing
                print("Buffer shape after reset:", ecg_buffer.shape)
                print("Copy Buffer shape:", copy_ecg_buffer.shape)
            
                model_output = get_prediction(ecg_data=copy_ecg_buffer)
                print("\n==============\nPrediction:", model_output)
                        
            physio_inference = {}
            if model_output:
                model_emotion_probs = model_output.get("probabilities")
                physio_inference = get_physiometrics(model_emotion_probs)
            
            behavioral_data = parse_game_state(game_state)
            if behavioral_data:
                # print_red("game_state: " + json.dumps(behavioral_data))
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
                global xai_agent_type
                if xai_agent_type == 'NoX':
                    save_metrics_data(behavioral_data, behavioral_data.get("playerId"), model_output)
                elif xai_agent_type == 'AdaX':
                    get_adax_explanation(behavioral_data, physio_inference, game_ts, model_output)
                    print(f"🧠 Inference time: {time.time() - start_time:.2f}s")
                elif xai_agent_type == 'StaticX':
                    get_static_explanation(behavioral_data, physio_inference, game_ts, model_output)
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


# === Step 4: Redirect `print()` to logging.info ===
class PrintLogger:
    def write(self, message):
        message = message.strip()
        if message:
            logger.info(message)

    def flush(self):
        pass  # Required for compatibility
    
def safe_thread(fn, name):
    def wrapper():
        try:
            fn()
        except Exception as e:
            logging.exception(f"❌ Thread '{name}' crashed: {e}")
    return wrapper

   
# ========== Start Everything ==========
if __name__ == '__main__':
    
    socket_connected.clear()
    threading.Thread(target=retry_socket_connection).start()
    ## old code below#
    # try:
    #     # connect to the socket server
    #     sio.connect(f'http://{SERVER_IP}:{PORT}')
    #     if not sio.connected:
    #         print_red("Socket connection failed. Retrying in 3s...")
    #         time.sleep(3)
    
    # except Exception as e:
    #     print_red("Socket Connection failed:" + e)
        
    # Start LSL stream listener
    threading.Thread(target=stream_ecg, daemon=True).start()
    threading.Thread(target=stream_eeg, daemon=True).start()
    threading.Thread(target=stream_metrics, daemon=True).start()
    threading.Thread(target=stream_game, daemon=True).start()
    # threading.Thread(target=run_lsl_logger, daemon=True).start()
    threading.Thread(target=process_and_explain, daemon=True).start()
    
    # # threading.Thread(target=safe_thread(stream_ecg, 'ECG'), daemon=True).start()
    # # threading.Thread(target=safe_thread(stream_eeg, 'EEG'), daemon=True).start()
    # # threading.Thread(target=safe_thread(stream_metrics, 'MET'), daemon=True).start()
    # threading.Thread(target=safe_thread(stream_game, 'OC'), daemon=True).start()
    # # threading.Thread(target=run_lsl_logger, daemon=True).start()
    # threading.Thread(target=safe_thread(process_and_explain, 'ADAX'), daemon=True).start()
    
    
    # #log
    
    if DEBUG_LOGGEER:
        # === Step 1: Create logs directory ===
        os.makedirs("app_logs", exist_ok=True)

        # === Step 2: Generate timestamped log filename ===
        log_filename = datetime.now().strftime("app_logs/session_%Y-%m-%d_%H-%M-%S.log")

        # === Step 3: Configure Logging ===
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        logger.handlers = []

        # File handler (writes all logs)
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler (prints to terminal)
        console_handler = logging.StreamHandler(sys.__stdout__)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        sys.stdout = PrintLogger()
        sys.stderr = PrintLogger()  # Optional: also redirect errors
    while True:
        time.sleep(1)  # Keep main thread alive

# Run as a background thread
