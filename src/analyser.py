import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from typing import List, Dict, Any
import ollama  # Using ollama's Python API directly

# Cache to store previously seen game states for deduplication
response_cache = {}

# Queue to handle incoming frame data
request_queue = deque()

# Constants
MAX_CONCURRENT = 4  # M2 supports multithreading but don’t overload
CACHE_EXPIRY = 30  # seconds

# Format prompt from game + physiological data
def format_prompt(data: Dict[str, Any]) -> str:
    return (
        f"At timestep {data.get('timestep')}, the AI was at position {data['players'][1]['position']} facing {data['players'][1]['orientation']}.\n"
        f"User state: stress {data.get('stress')}, trust {data.get('trust')}, cognitive load {data.get('cognitive_load')}.\n"
        f"Current joint action: {data.get('joint_action')}. Score: {data.get('score')}, Collisions: {data.get('num_collisions')}, Time left: {data.get('time_left')}s.\n"
        f"Orders: {data.get('all_orders')}. Objects: {data.get('objects')}.\n"
        f"Explain the AI’s likely behavior and optionally predict its next action."
    )

# Create a unique key for caching
def hash_state(data: Dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()

# Chat wrapper (faster call)
def ollama_chat_single(prompt: str, model="mistral:7b-instruct-q4_K_M", temperature=0.3, max_tokens=200):
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Main processing loop

def process_requests():
    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = []
        while request_queue:
            item = request_queue.popleft()
            state_hash = item["hash"]

            # Skip if already in cache
            if state_hash in response_cache:
                if time.time() - response_cache[state_hash]["timestamp"] < CACHE_EXPIRY:
                    results.append((item["timestep"], response_cache[state_hash]["response"]))
                    continue

            futures.append(executor.submit(process_single_item, item))

        for future in as_completed(futures):
            if future.result():
                results.append(future.result())

    print(f"Processed in {round((time.time() - start)*1000, 2)}ms")
    return results

# Worker for a single record

def process_single_item(item):
    prompt = item["prompt"]
    state_hash = item["hash"]
    timestep = item["timestep"]

    response = ollama_chat_single(prompt)

    response_cache[state_hash] = {
        "response": response,
        "timestamp": time.time()
    }
    return (timestep, response)

# Example usage:

def enqueue_frame_data(frame_data: Dict[str, Any]):
    prompt = format_prompt(frame_data)
    state_hash = hash_state(frame_data)
    request_queue.append({
        "timestep": frame_data.get("timestep", -1),
        "prompt": prompt,
        "hash": state_hash
    })

# Simulated frame input
# enqueue_frame_data(game_frame_data)
# process_requests()
