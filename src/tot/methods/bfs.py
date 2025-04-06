import itertools
import numpy as np
from functools import partial
from tot.models import gpt, local_model
from tot.tasks.adax import AdaXTask
import json

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = local_model(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample, is_local):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    if is_local:
        vote_outputs = local_model(vote_prompt, n=n_evaluate_sample, stop=None)
    else:    
        vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = local_model(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]


def summarize_rag_rows(df, top_n=3):
    """
    Summarizes top-N RAG result rows with structured game/user state and explanation metadata.
    
    Parameters:
        df (pd.DataFrame): RAG dataframe with fields like 'content', 'task_description', etc.
        top_n (int): Number of top rows to include based on lowest distance.

    Returns:
        str: Multi-line string summary.
    """
    df = df.dropna(subset=["content"]) #.sort_values(by="distance").head(top_n)
    print("=======\n", df, "\n=======")
    summaries = []
    for _, row in df.iterrows():
        timestamp = row.get("created_at", "Recent time")
        task = row.get("task_description", "Understanding AI's latest decision")
        content = row["content"].replace("\n", " ").strip()

        # Extract info from content if needed
        game_state = f"Score {row.get('score', '?')}, {row.get('num_collisions', '?')} collisions, {row.get('time_left', '?')} seconds left"
        user_state = f"Stress {row.get('stress', '?')}, trust {row.get('trust', '?')}, cognitive load {row.get('cognitive_load', '?')}"
        explanation_features = row.get("explanation_features", {})
        explanation = row.get("final_explanation", "?")
        has_features = False
        if row.get("explanation_features", ""):
            has_features = True
            duration = row.get("explanation_features", {}).get("explanation_duration", "?")
            granularity = row.get("explanation_features", {}).get("explanation_granularity", "?")
            timing = row.get("explanation_features", {}).get("explanation_timing", "?")
        if has_features:
            summary = (
                f"At {timestamp}, the game state was: {game_state}. "
                f"User state: {user_state}. "
                # f"Task: {task}. "
                # f"Explanation: {explanation} "
                f"Explanatory features: {explanation_features}. "
                f"(Duration: {duration}, Granularity: {granularity}, Timing: {timing})."
            )
        else:
            summary = (
                f"At {timestamp}, the game state was: {game_state}. "
                f"User state: {user_state}. "
                # f"Task: {task}. "
                # f"Explanation: {explanation} "
                f"Explanatory features: {explanation_features}. "
            )
            
        summaries.append(summary)


def get_recent_context(x, vector_db, return_dataframe=True, xai_agent_type='StaticX'):
    input_dict = json.loads(x)
    prev_context = {}
    print("==xai_agent_type==", xai_agent_type)
    if xai_agent_type == 'AdaX':
        recent_context = AdaXTask.prepare_prev_context(input_dict, vector_db, return_dataframe=return_dataframe)
        print("recent_context", recent_context)
        prev_context = summarize_rag_rows(recent_context)
    curr_game_summary = AdaXTask.summarize_current_game(input_dict)
    print("------------prep_context-----\n",prev_context, "\n\n")
    
    return { "prev": prev_context, "current": curr_game_summary}

def get_samples(task, x, y, context, n_generate_sample, prompt_sample, stop, is_local):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y, context)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y, context)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    import time
    start = time.time()
    print(f"\nPrompt: {prompt}\n\n")
    if is_local:
        samples = local_model(prompt, n=n_generate_sample, stop=stop) # Generate n samples for the prompt
    else:
        samples = gpt(prompt, n=n_generate_sample, stop=stop) # Generate n samples for the prompt
    end = time.time()
    print(f"Time taken to generate samples: {end-start}")
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True, vector_db=None, parallel=False, return_dataframe=True, xai_agent_type='StaticX'):
    if args.is_local:
        global local_model
        local_model = partial(local_model, model=args.backend, temperature=args.temperature) #TODO: add parallel option
        print(local_model)
    else:
        global gpt
        gpt = partial(gpt, model=args.backend, temperature=args.temperature, parallel=parallel)
        print(gpt)
    x = task.get_input(idx)  # input
    context = get_recent_context(x, vector_db, return_dataframe=return_dataframe, xai_agent_type=args.xai_agent_type)
    
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, context, args.n_generate_sample, args.prompt_sample, stop=task.stops[step], is_local=args.is_local) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample, is_local=args.is_local)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        #TODO: print adax features
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'\n-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(f'\n-- final choices --: {ys}\n')
    #TODO: save current state and selected adax features
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global local_model
    local_model = partial(local_model, model=args.backend, temperature=args.temperature)
    print(local_model)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}