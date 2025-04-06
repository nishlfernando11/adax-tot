# main_benchmark.py
from openai import OpenAI
from transformers import pipeline
from dotenv import load_dotenv
import os, time, requests

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def benchmark_openai(model, prompt):
    start = time.time()
    try:
        res = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = time.time() - start
        return res.choices[0].message.content.strip(), latency
    except Exception as e:
        return f"[OpenAI error] {e}", -1

def benchmark_ollama(model, prompt):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False}
    start = time.time()
    try:
        r = requests.post("http://localhost:11434/api/chat", json=payload)
        latency = time.time() - start
        return r.json()["message"]["content"].strip(), latency
    except Exception as e:
        return f"[Ollama error] {e}", -1

def benchmark_hf(model_id, prompt):
    start = time.time()
    try:
        pipe = pipeline("text-generation", model=model_id)
        output = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
        latency = time.time() - start
        return output.strip(), latency
    except Exception as e:
        return f"[HF error] {e}", -1

if __name__ == "__main__":
    
    prompt = '''
    You are observing the current Overcooked game session between an AI chef and a human chef.

    Based on the following real-time data, generate an adaptive explanation (≤10 words) for the AI chef’s current behavior.

    **User’s Current Physiological and Emotional State**: {'cognitive_load': 'low', 'stress': 'high'}
    **User’s Current Task-Behavioral State**:{'state': '{"players": [{"position": [1, 2], "orientation": [0, -1], "held_object": null}, {"position": [3, 2], "orientation": [1, 0], "held_object": null}], "objects": [], "bonus_orders": [], "all_orders": [{"ingredients": ["onion", "onion", "onion"]}], "timestep": 34}', 'joint_action': '[[0, 0], [0, -1]]', 'reward': 0, 'time_left': 23.78111743927002, 'score': 0, 'time_elapsed': 6.2188825607299805, 'cur_gameloop': 35, 'layout': '[["X", "X", "P", "X", "X"], ["O", " ", " ", " ", "O"], ["X", " ", " ", " ", "X"], ["X", "D", "X", "S", "X"]]', 'layout_name': 'cramped_room', 'trial_id': '1742539546.4407632', 'player_0_id': '74994c5fd6de44f399efe56ea6925b4e', 'player_1_id': 'PPOCrampedRoom_1', 'player_0_is_human': True, 'player_1_is_human': False, 'collision': False, 'num_collisions': 2, 'unix_timestamp': '1742539552.6596458'}
    **Recent User-AI Interaction Context**:At 2025-03-24T00:09:21.755248, the game state was: Score 50, 0 collisions, 22 seconds left. User state: Stress medium, trust low, cognitive load high. Task: Understanding AI's latest decision. Explanation: Delivered completed dish. AI finished meal preparation and ensured timely delivery. 1. Checked order readiness. 2. Picked the dish. 3. Moved to serving area. 4. Delivered to counter. Explanatory features: {'explanation_timing': 'proactive', 'explanation_duration': 'long', 'explanation_granularity': 'highlevel'}. 

    At 2025-03-24T00:09:24.593104, the game state was: Score 45, 5 collisions, 34 seconds left. User state: Stress medium, trust medium, cognitive load medium. Task: Understanding AI's latest decision. Explanation: Took a different path to avoid collision. AI adjusted its path dynamically to avoid congestion and improve efficiency. 1. Detected potential collision. 2. Adjusted path. 3. Found alternative route. 4. Reached ingredient station. Explanatory features: {'explanation_timing': 'reactive', 'explanation_duration': 'short', 'explanation_granularity': 'steps'}. 

    At 2025-03-24T00:09:15.900020, the game state was: Score 15, 10 collisions, 57 seconds left. User state: Stress high, trust high, cognitive load high. Task: Understanding AI's latest decision. Explanation: Moved to chop station for onion preparation. AI prioritized ingredient preparation and moved to the chop station for cutting onions. 1. Identified missing ingredient. 2. Moved to chop station. 3. Chopped onion. 4. Brought it to cooking pot. Explanatory features: {'explanation_timing': 'proactive', 'explanation_duration': 'medium', 'explanation_granularity': 'highlevel'}. 

    State whether the assistant has enough context to answer the question:
    - **Yes, the assistant has enough context.**
    - **No, the assistant needs more context.**

    **Requirements**:
    - The explanation must be concise, meaningful, and relevant to the AI’s current decision.
    - It must **adapt** to the user's emotional and behavioral states.
    - Avoid generalizations. Be specific to the scenario.
    - Justify or reassure if user state is poor (e.g., high stress or low trust).
    - Include explanation features in the following format:
        "features": ["duration: short", "granularity: detailed", "timing: reactive"]
    - Output in JSON format:
        # {
        #    "answer": "...",
        #    "features": ["...", "..."],
        #    "enough_context": true/false
        # }
    '''
    results = []

    # 🔹 OpenAI Models
    for model in ["gpt-3.5-turbo", "gpt-4"]:
        out, t = benchmark_openai(model, prompt)
        results.append(("openai", model, t, out))

    # 🔹 Ollama Models
    ollama_models = ["mistral", "llama2", "codellama", "dolphin-mistral"]
    for model in ollama_models:
        out, t = benchmark_ollama(model, prompt)
        results.append(("ollama", model, t, out))

    # 🔹 Hugging Face Model
    hf_models = ["tiiuae/falcon-7b-instruct"]
    for model in hf_models:
        out, t = benchmark_hf(model, prompt)
        results.append(("huggingface", model, t, out))

    # Print Summary
    print("\n🔎 Benchmark Summary:")
    for backend, model, latency, output in results:
        print(f"[{backend}] {model}: {latency:.2f}s -> {output[:80]}...")
