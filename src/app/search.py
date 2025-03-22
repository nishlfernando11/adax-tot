from datetime import datetime, timedelta
from app.database.vector_store import VectorStore
from app.services.synthesizer_mistral import Synthesizer
from timescale_vector import client
import pandas as pd
import json

# Initialize VectorStore
vec = VectorStore()
def extract_system_context(state):
    """
    Extracts key user-AI interaction elements from any system.

    Parameters:
    - system_state (dict): AI system state including user metrics & AI decisions.

    Returns:
    - (dict): Extracted general features applicable across domains.
    """
    return {
        # "ai_action": system_state.get("ai_action", "unknown"),
        # "user_action": system_state.get("user_action", "unknown"),
        # "ai_decision_confidence": system_state.get("ai_decision_confidence", "="),
        # "system_performance": system_state.get("system_performance", 0),
        # "risk_perception": system_state.get("risk_perception", ""),
        "stress": state["physiological_state"]["stress"],
        "trust": state["physiological_state"]["trust"], 
        "cognitive_load": state["physiological_state"]["cognitive_load"],
        "num_collisions": state["behavioral_state"]["num_collisions"],
        "score": state["behavioral_state"]["score"],
        "time_left": state["behavioral_state"]["time_left"]
    }

def infer_task_description(context):
    #TODO: Add more generalized task types
    """
    Dynamically infers a high-level AI task type based on system context.

    Parameters:
    - context (dict): Extracted system context.

    Returns:
    - (str): The inferred generalized task description.
    """
    # ai_action = context["ai_action"]
    # user_action = context["user_action"]
    # trust = context["trust"]
    # stress = context["stress"]
    # risk_perception = context["risk_perception"]

    # Generalized task types
    # if "prioritize" in ai_action:
    #     return "Understanding AI's decision-making priority"

    # if "move" in ai_action and risk_perception in ["high", "uncertain"]:
    #     return "Understanding why AI took a safer route"

    # if "wait" in ai_action and stress == "high":
    #     return "Understanding why AI delayed action to reduce overload"

    # if trust == "low":
    #     return "Understanding AI's reasoning to improve trust"

    return "Understanding AI's latest decision"

def get_context_features(state):
    # Extract context
    context_features = extract_system_context(state)

    # Dynamically infer task description
    task_description = infer_task_description(context_features)

    # Current user and game state
    current_state = {
        "stress": context_features["stress"],
        "trust": context_features["trust"],
        "cognitive_load": context_features["cognitive_load"],
        "score": context_features["score"],
        "num_collisions": context_features["num_collisions"],
        "time_left": context_features["time_left"],
        "task_description": task_description,  # Auto-inferred task description
    }
    
    return current_state

def get_recent_context(current_state):
    # Retrieve **recent interactions** (limit to last n period)
    # @TODO: Replace with actual time range
    time_range = (datetime.now() - timedelta(days=20), datetime.now()) # TODO:make 10 secs, or accept time range as input
    query = format_content(current_state)

    # ✅ Fix: Provide query_text for similarity search
    recent_context = vec.search(
        query_text= query, #current_state["task_description"],  # ✅ Required argument
        limit=3,
        # predicates=predicates,
        time_range=time_range
    )
    # Convert `recent_context` into a DataFrame if it's a list of dicts
    recent_context_df = pd.DataFrame(recent_context.loc[:, recent_context.columns != 'embedding'])
    return recent_context_df

def get_context_df(current_state, recent_context_df):
    # context_data = {
    # "task_context": current_state["task_description"],
    # "stress": current_state["stress"],
    # "trust": current_state["trust"],
    # "cognitive_load": current_state["cognitive_load"],
    # "score": current_state["score"],
    # "num_collisions": current_state["num_collisions"],
    # }
    current_state_df = pd.DataFrame([current_state])

    context_df = pd.concat([current_state_df, recent_context_df], ignore_index=True)
    print(context_df)
    # Changed to return recent  state
    return recent_context_df



def format_content(state):
    '''
    Formats the content of a state into a string for similarity search.'''
    content = (
        f"Game state: Score {state['score']}, {state['num_collisions']} collisions,\n"
        f"{state['time_left']} seconds left.\n"
        f"User state: Stress {state['stress']}, trust {state['trust']}, cognitive load {state['cognitive_load']}.\n"
        f"Task: {state['task_description']}\n"
        # f"Best Explanation: {state['explanation_simplified']}\n"
        # f"Alternative Explanation (Balanced): {state['explanation_balanced']}\n"
        # f"Step-by-Step Explanation: {state['explanation_step_by_step']}\n"
        # f"duration: {state.get('explanation_duration', 'medium')},  # Default: 'medium'\n"
        # f"granularity: {state.get('explanation_granularity', 'steps')},  # Default: 'steps'\n"
        # f"timing: {state.get('explanation_timing', 'reactive')}  # Default: 'reactive'"
    )
    return content
