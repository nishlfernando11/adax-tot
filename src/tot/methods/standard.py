import itertools
import numpy as np
from functools import partial
from tot.models import gpt, local_model
from tot.tasks.adax import AdaXTask
import json
import difflib
import psycopg2
from psycopg2.extras import Json
from statistics import mean
import re
from collections import deque

RECENT_STATE_SUMMARY_QUEUE = deque(maxlen=5)  # auto-removes oldest entries
USE_CACHE = False
NUM_ITERATIONS = 0
CACHE_LIMIT = 2

# ------- #
# Mistral #
# ------- #

# Static knowledge base for explanation rules
STATIC_KNOWLEDGE_BASE = """
# Role and Purpose:
You are an AI assistant specializing in adaptive explainability for Human-Machine Teaming (HMT).
Generate one-sentence explanations (≤10 words) for AI actions in Overcooked.

# Constraints:
- Adapt to user's stress, trust, cognitive load.
- Tailor explanations by game metrics (score, collisions).
- Maintain clarity under pressure.
- Each explanation must include adaptive features: [duration, granularity, timing].
- Justify explanation type chosen.
"""

def genAdaX(args, task, idx, is_local=False):
    x = task.get_input(idx)  # input
    return get_explanation(x, args, is_local)
    

def get_recent_states(input_dict):
    prev_state_summary = AdaXTask.prepare_context(input_dict)
    prev_state_summary = summarize_rag_rows(prev_state_summary)
    
    return prev_state_summary

def get_explanation(x, args, is_local):
    global USE_CACHE
    global NUM_ITERATIONS
    global CACHE_LIMIT
    input_dict = json.loads(x)
    print("USE_CACHE===>?", USE_CACHE)
    print("NUM_ITERATIONS===>?", NUM_ITERATIONS)
    state_summary = summarize_state(input_dict)
    print("state_summary===>?", state_summary)
    if USE_CACHE:        
        prev_state_summary = list(RECENT_STATE_SUMMARY_QUEUE)[-4:-1]
        print("RECENT_STATE_SUMMARY_QUEUE===>?", prev_state_summary)
        
        USE_CACHE = False
    else:
        prev_state_summary = get_recent_states(input_dict)
        print("prev_state_summary===>?", prev_state_summary)
        print("NUM_ITERATIONS > CACHE_LIMIT===>?", NUM_ITERATIONS > CACHE_LIMIT)    

        if NUM_ITERATIONS > CACHE_LIMIT:
            USE_CACHE = True
            NUM_ITERATIONS = 0
        NUM_ITERATIONS += 1
    return generate_adaptive_explanation(state_summary, prev_state_summary, args, is_local)
    
def generate_adaptive_explanation(current_state, recent_states, args, is_local, kb=STATIC_KNOWLEDGE_BASE):
    """
    Generate adaptive explanations based on dynamic user and game states.

    Parameters:
        user_state (dict): Current physiological/emotional states (e.g., stress, trust, cognitive_load).
        task_state (dict): Current game metrics (score, num_collisions, time_left).
        recent_states (list): List of recent user/task states and explanations.
        kb (str): Knowledge base content (static instructions for explanation selection).
        model_name (str): Name of the model to use for inference.
        n_generate_sample (int, optional): Number of generated samples. Defaults to 1.
        stop (list, optional): Stop tokens for model generation. Defaults to None.

    Returns:
        dict: Generated explanation and adaptive features.
    """

    prompt = f"""
    {kb}

    Current State summary: {current_state}
    Recent States summary: {recent_states}

    Generate an explanation (max 10 words) for AI chef's action based on state analysis. Use features to adapt to user/game context aiming to improve
    user collaboration, performance and trust and minimise conflicts.
    Provide adaptive features as well.

    State whether the assistant has enough context to answer the question:
    - **Yes, the assistant has enough context.**
    - **No, the assistant needs more context.**
    
    - Output in strictly JSON format. :
        {{
        "answer": "Your concise answer here.",
        "justification": "justify the explanation and feature selection.",
        "features": {{
            "duration": "short/long",
            "granularity": "highlevel/detailed",
            "timing": "proactive/reactive"
        }},
        "enough_context": true/false
        }}
    Explanation:    
    """
    print("prompt===>?", prompt)
    global running_model
    if is_local:
        running_model = partial(local_model, model=args.backend, temperature=args.temperature)
    else:
        running_model = partial(gpt, model=args.backend, temperature=args.temperature)
    
    print(running_model)
    
    generated_text = running_model(prompt, n=args.n_generate_sample, stop=args.stop)[0].strip()
    print("generated_text===>?", generated_text)
    # Parse the generated text into explanation and features
    # explanation_lines = json.loads(generated_text)
    # Safely extract JSON using regex
    if not generated_text:
        return None
    decoded_response = json.loads(generated_text)
    return decoded_response
    # json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
    # if json_match:
    #     json_str = json_match.group()
    #     decoded_response = json.loads(json_str)
    #     print(decoded_response)
    #     return decoded_response
    # else:
    #     print("No valid JSON found.")
    #     return None
    # explanation = explanation_lines[0].replace('Explanation:', '').strip()

# Example usage
if __name__ == '__main__':
    user_state = {'stress': 'medium', 'trust': 'low', 'cognitive_load': 'medium'}
    task_state = {'score': 19, 'num_collisions': 3, 'time_left': 4}
    recent_states = [
        {'timestamp': '2025-03-22T18:57:55', 'game_state': {'score': 45, 'num_collisions': 5}, 'user_state': {'stress': 'medium', 'trust': 'medium', 'cognitive_load': 'medium'}, 'explanation': 'Took a different path to avoid collision.'},
        {'timestamp': '2025-03-22T18:57:50', 'game_state': {'score': 50, 'num_collisions': 0}, 'user_state': {'stress': 'medium', 'trust': 'low', 'cognitive_load': 'high'}, 'explanation': 'Delivered completed dish.'}
    ]

    kb = "Your detailed KB/system instructions here (copied from your prompt)"

    model_name = 'mistral'

    result = generate_adaptive_explanation(user_state, task_state, recent_states, kb, model_name)

    print("Adaptive Explanation:", result["explanation"])
    print("Adaptive Features:", result["features"])
# ------- #

# Agent to summarize the current user + game state into a natural language string

def summarize_state(input_dict):
    print("input_dict===>", input_dict)
    try:
        # Extract physiological and behavioral states
        physiological = input_dict.get("physiological_state", {})
        behavioral = input_dict.get("behavioral_state", {})

        # Format user state string
        user_state = (
            f"User state: stress {physiological.get('stress', '?')}, "
            # f"trust {physiological.get('trust', '?')}, "
            f"cognitive load {physiological.get('cognitive_load', '?')}"
        )

        # Parse JSON string of game state
        game_state_raw = behavioral.get("state", '{}')
        game_state = json.loads(game_state_raw)

        # Extract player and environment data
        ai = game_state['players'][1]
        user = game_state['players'][0]

        ai_position = ai.get("position", '?')
        ai_orientation = ai.get("orientation", '?')
        held_object = ai.get("held_object", 'none')
        teammate_position = user.get("position", '?')

        orders = game_state.get("all_orders", [])
        order_ingredients = ', '.join(orders[0].get("ingredients", [])) if orders else "none"

        score = behavioral.get("score", '?')
        collisions = behavioral.get("num_collisions", '?')
        time_left = behavioral.get("time_left", '?')

        game_state_str = (
            f"Game state: score {score}, {collisions} collisions, {time_left} seconds left."
        )

        # Trend analysis (if any)
        trend_info = analyze_trends(input_dict.get("recent_states", []))

        # Additional state info
        layout_name = behavioral.get("layout_name", '?')
        timestep = game_state.get("timestep", '?')
        cur_gameloop = behavioral.get("cur_gameloop", '?')
        joint_action = behavioral.get("joint_action", '?')
        collision_flag = behavioral.get("collision", '?')
        reward = behavioral.get("reward", '?')

        # Build full summary
        full_state = (
            f"AI position: {ai_position}, facing {ai_orientation}, holding: {held_object}. "
            f"Teammate position: {teammate_position}. "
            f"Orders require: {order_ingredients}. "
            f"Layout: {layout_name}, timestep: {timestep}, gameloop: {cur_gameloop}. "
            f"Last joint action: {joint_action}, reward: {reward}, collision occurred: {collision_flag}."
        )

        analysis_summary = f"{user_state}. {game_state_str} {trend_info} {full_state}"
        return analysis_summary

    except Exception as e:
        return f"State summary error: {str(e)}"
    
    # print("input_dict===>?", input_dict)
    # try:
    #     user_state = f"User state: stress {input_dict['physiological_state']['stress']}, trust {input_dict['physiological_state']['trust']}, cognitive load {input_dict['physiological_state']['cognitive_load']}"
    #     game_state = f"Game state: score {input_dict['behavioral_state']['score']}, {input_dict['behavioral_state']['num_collisions']} collisions, {input_dict['behavioral_state']['time_left']} seconds left."
    #     trend_info = analyze_trends(input_dict.get("recent_states", []))

    #     ai_position = input_dict.get("ai_position", '?')
    #     ai_orientation = input_dict.get("ai_orientation", '?')
    #     held_object = input_dict.get("ai_held_object", 'none')
    #     teammate_position = input_dict.get("teammate_position", '?')
    #     station_states = input_dict.get("station_states", {})
    #     order_ingredients = ', '.join(input_dict.get("orders", [{}])[0].get("ingredients", [])) if input_dict.get("orders") else "none"

    #     layout_name = input_dict.get("layout_name", '?')
    #     timestep = input_dict.get("timestep", '?')
    #     cur_gameloop = input_dict.get("cur_gameloop", '?')
    #     joint_action = input_dict.get("joint_action", '?')
    #     collision_flag = input_dict.get("collision", '?')
    #     reward = input_dict.get("reward", '?')

    #     station_info = f"Station states: {json.dumps(station_states)}. " if station_states else ""

    #     full_state = (
    #         f"AI position: {ai_position}, facing {ai_orientation}, holding: {held_object}. "
    #         f"Teammate position: {teammate_position}. "
    #         f"Orders require: {order_ingredients}. "
    #         f"Layout: {layout_name}, timestep: {timestep}, gameloop: {cur_gameloop}. "
    #         f"Last joint action: {joint_action}, reward: {reward}, collision occurred: {collision_flag}. "
    #         f"{station_info}"
    #     )
    #     analysis_summary = f"{user_state}. {game_state} {trend_info} {full_state}"
    #     RECENT_STATE_SUMMARY_QUEUE.append(analysis_summary)
    #     return analysis_summary

    # except Exception as e:
    #     return f"State summary error: {str(e)}"

# Basic trend analyzer for cognitive/game metrics
def analyze_trends(recent_states):
    if not recent_states or len(recent_states) < 2:
        return ""
    try:
        scores = [s.get("score", 0) for s in recent_states]
        collisions = [s.get("num_collisions", 0) for s in recent_states]
        score_trend = "increasing" if scores[-1] > mean(scores[:-1]) else "decreasing"
        collision_trend = "rising" if collisions[-1] > mean(collisions[:-1]) else "stable"
        return f"Trend: score is {score_trend}, collisions are {collision_trend}."
    except:
        return ""

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = mistral_local(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:
        if y in local_value_cache:
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = mistral_local(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = mistral_local(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]


def summarize_rag_rows(df, top_n=3):
    df = df.dropna(subset=["content"])
    summaries = []
    for _, row in df.iterrows():
        timestamp = row.get("created_at", "Recent time")
        task = row.get("task_description", "Understanding AI's latest decision")
        content = row["content"].replace("\n", " ").strip()
        game_state = f"Score {row.get('score', '?')}, {row.get('num_collisions', '?')} collisions, {row.get('time_left', '?')} seconds left"
        user_state = f"Stress {row.get('stress', '?')}, trust {row.get('trust', '?')}, cognitive load {row.get('cognitive_load', '?')}"
        explanation = row.get("final_explanation", "?")
        explanation_features = row.get("explanation_features", {})
        summary = (
            f"At {timestamp}, the game state was: {game_state}. "
            f"User state: {user_state}. "
            f"Task: {task}. "
            f"Explanation: {explanation} "
            f"Explanatory features: {explanation_features}."
        )
        RECENT_STATE_SUMMARY_QUEUE.append(summary)
        summaries.append(summary)
    return "\n\n".join(summaries)

def get_recent_context(x):
    input_dict = json.loads(x)
    state_summary = summarize_state(input_dict)
    context_df = AdaXTask.prepare_context(input_dict)
    context = summarize_rag_rows(context_df)
    return STATIC_KNOWLEDGE_BASE + "\n\n" + "### SYSTEM CONTEXT:\n" + state_summary + "\n\n" + context

def deduplicate_thoughts(thoughts, threshold=0.85):
    final = []
    for t in thoughts:
        if all(difflib.SequenceMatcher(None, t, existing).ratio() < threshold for existing in final):
            final.append(t)
    return final

def get_samples(task, x, y, context, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y, context)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y, context)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    import time
    start = time.time()
    print(f"Prompt: {prompt}")
    samples = mistral_local(prompt, n=n_generate_sample, stop=stop)
    print(f"Time taken to generate samples: {round((time.time() - start) * 1000, 2)} ms")
    return [y + _ for _ in samples]

def log_explanation_to_db(explanation, features, context, conn_params):
    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO explanation_logs (timestamp, explanation, features, context)
            VALUES (NOW(), %s, %s, %s);
        """, (explanation, Json(features), context))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB log error: {e}")

def solve(args, task, idx, to_print=True, db_conn_params=None):
    global mistral_local
    mistral_local = partial(mistral_local, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)
    context = get_recent_context(x)
    ys = ['']
    infos = []
    for step in range(task.steps):
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, context, args.n_generate_sample, args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        new_ys = deduplicate_thoughts(new_ys)
        ids = list(range(len(new_ys)))
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'\n-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
        # if db_conn_params:
        #     log_explanation_to_db(ys[0], {'score': values[select_ids[0]]}, context, db_conn_params)
    if to_print:
        print(f'\n-- final choices --: {ys}\n')
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global mistral_local
    mistral_local = partial(mistral_local, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}


#-------#

# def get_value(task, x, y, n_evaluate_sample, cache_value=True):
#     value_prompt = task.value_prompt_wrap(x, y)
#     if cache_value and value_prompt in task.value_cache:
#         return task.value_cache[value_prompt]
#     value_outputs = mistral_local(value_prompt, n=n_evaluate_sample, stop=None)
#     value = task.value_outputs_unwrap(x, y, value_outputs)
#     if cache_value:
#         task.value_cache[value_prompt] = value
#     return value

# def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
#     values = []
#     local_value_cache = {}
#     for y in ys:  # each partial output
#         if y in local_value_cache:  # avoid duplicate candidates
#             value = 0
#         else:    
#             value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
#             local_value_cache[y] = value
#         values.append(value)
#     return values

# def get_votes(task, x, ys, n_evaluate_sample):
#     vote_prompt = task.vote_prompt_wrap(x, ys)
#     vote_outputs = mistral_local(vote_prompt, n=n_evaluate_sample, stop=None)
#     values = task.vote_outputs_unwrap(vote_outputs, len(ys))
#     return values

# def get_proposals(task, x, y): 
#     propose_prompt = task.propose_prompt_wrap(x, y)
#     proposals = mistral_local(propose_prompt, n=1, stop=None)[0].split('\n')
#     return [y + _ + '\n' for _ in proposals]


# def summarize_rag_rows(df, top_n=3):
#     """
#     Summarizes top-N RAG result rows with structured game/user state and explanation metadata.
    
#     Parameters:
#         df (pd.DataFrame): RAG dataframe with fields like 'content', 'task_description', etc.
#         top_n (int): Number of top rows to include based on lowest distance.

#     Returns:
#         str: Multi-line string summary.
#     """
#     df = df.dropna(subset=["content"]) #.sort_values(by="distance").head(top_n)
#     print("=======\n", df, "\n=======")
#     summaries = []
#     for _, row in df.iterrows():
#         timestamp = row.get("created_at", "Recent time")
#         task = row.get("task_description", "Understanding AI's latest decision")
#         content = row["content"].replace("\n", " ").strip()

#         # Extract info from content if needed
#         game_state = f"Score {row.get('score', '?')}, {row.get('num_collisions', '?')} collisions, {row.get('time_left', '?')} seconds left"
#         user_state = f"Stress {row.get('stress', '?')}, trust {row.get('trust', '?')}, cognitive load {row.get('cognitive_load', '?')}"
#         explanation_features = row.get("explanation_features", {})
#         explanation = row.get("final_explanation", "?")
#         duration = row.get("explanation_duration", "?")
#         granularity = row.get("explanation_granularity", "?")
#         timing = row.get("explanation_timing", "?")

#         summary = (
#             f"At {timestamp}, the game state was: {game_state}. "
#             f"User state: {user_state}. "
#             f"Task: {task}. "
#             f"Explanation: {explanation} "
#             f"Explanatory features: {explanation_features}. "
#             # f"(Duration: {duration}, Granularity: {granularity}, Timing: {timing})."
#         )
#         summaries.append(summary)

#     return "\n\n".join(summaries)


# def get_recent_context(x):
#     input_dict = json.loads(x)
#     context_df = AdaXTask.prepare_context(input_dict)
#     prep_context = summarize_rag_rows(context_df)
#     return prep_context

# def get_samples(task, x, y, context, n_generate_sample, prompt_sample, stop):
#     if prompt_sample == 'standard':
#         prompt = task.standard_prompt_wrap(x, y, context)
#     elif prompt_sample == 'cot':
#         prompt = task.cot_prompt_wrap(x, y, context)
#     else:
#         raise ValueError(f'prompt_sample {prompt_sample} not recognized')
#     import time
#     start = time.time()
#     print(f"Prompt: {prompt}")
#     samples = mistral_local(prompt, n=n_generate_sample, stop=stop) # Generate n samples for the prompt
#     end = time.time()
#     print(f"Time taken to generate samples: {end-start}")
#     return [y + _ for _ in samples]

# def solve(args, task, idx, to_print=True):
#     global mistral_local
#     mistral_local = partial(mistral_local, model=args.backend, temperature=args.temperature)
#     print(mistral_local)
#     x = task.get_input(idx)  # input
#     context = get_recent_context(x)
    
#     ys = ['']  # current output candidates
#     infos = []
#     for step in range(task.steps):
#         # generation
#         if args.method_generate == 'sample':
#             new_ys = [get_samples(task, x, y, context, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
#         elif args.method_generate == 'propose':
#             new_ys = [get_proposals(task, x, y) for y in ys]
#         new_ys = list(itertools.chain(*new_ys))
#         ids = list(range(len(new_ys)))
#         # evaluation
#         if args.method_evaluate == 'vote':
#             values = get_votes(task, x, new_ys, args.n_evaluate_sample)
#         elif args.method_evaluate == 'value':
#             values = get_values(task, x, new_ys, args.n_evaluate_sample)

#         # selection
#         if args.method_select == 'sample':
#             ps = np.array(values) / sum(values)
#             select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
#         elif args.method_select == 'greedy':
#             select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
#         select_new_ys = [new_ys[select_id] for select_id in select_ids]

#         # log
#         #TODO: print adax features
#         if to_print: 
#             sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
#             print(f'\n-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
#         infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
#         ys = select_new_ys
    
#     if to_print: 
#         print(f'\n-- final choices --: {ys}\n')
#     #TODO: save current state and selected adax features
#     return ys, {'steps': infos}

# def naive_solve(args, task, idx, to_print=True):
#     global mistral_local
#     mistral_local = partial(mistral_local, model=args.backend, temperature=args.temperature)
#     print(mistral_local)
#     x = task.get_input(idx)  # input
#     ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
#     return ys, {}