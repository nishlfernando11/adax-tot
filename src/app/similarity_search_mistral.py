from datetime import datetime, timedelta
from app.database.vector_store import VectorStore
from app.services.synthesizer_mistral import Synthesizer
from timescale_vector import client
import pandas as pd
import json

# Initialize VectorStore
vec = VectorStore()
def extract_system_context(system_state):
    """
    Extracts key user-AI interaction elements from any system.

    Parameters:
    - system_state (dict): AI system state including user metrics & AI decisions.

    Returns:
    - (dict): Extracted general features applicable across domains.
    """
    return {
        "ai_action": system_state.get("ai_action", "unknown"),
        "user_action": system_state.get("user_action", "unknown"),
        "ai_decision_confidence": system_state.get("ai_decision_confidence", "="),
        "system_performance": system_state.get("system_performance", 0),
        "risk_perception": system_state.get("risk_perception", ""),
        "stress": system_state.get("stress", ""),
        "trust": system_state.get("trust", ""),
        "cognitive_load": system_state.get("cognitive_load", ""),
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

# Simulated incoming real-time system state
system_state = {
    "ai_action": "move_toward_station",
    "user_action": "prepare_ingredient",
    "ai_decision_confidence": "high",
    "system_performance": 85,
    "risk_perception": "low",
    "stress": "medium",
    "trust": "low",
    "cognitive_load": "high",
}


def get_context_features(system_state):
    # Extract context
    context_features = extract_system_context(system_state)

    # Dynamically infer task description
    task_description = infer_task_description(context_features)

    # Current user and game state
    current_state = {
        "stress": context_features["stress"],
        "trust": context_features["trust"],
        "cognitive_load": context_features["cognitive_load"],
        "game_score": context_features["system_performance"],  # Using performance as score
        "num_collisions": system_state.get("num_collisions", 0),
        "task_description": task_description,  # Auto-inferred task description
    }
    
    return current_state

def get_recent_context(current_state):
    # Retrieve **recent interactions** (limit to last n period)
    # @TODO: Replace with actual time range
    time_range = (datetime.now() - timedelta(days=20), datetime.now()) # make 10 secs

    # ✅ Fix: Provide query_text for similarity search
    recent_context = vec.search(
        query_text=current_state["task_description"],  # ✅ Required argument
        limit=3,
        # predicates=predicates,
        time_range=time_range
    )
    # Convert `recent_context` into a DataFrame if it's a list of dicts
    recent_context_df = pd.DataFrame(recent_context)
    return recent_context_df

def get_context_df(current_state, recent_context_df):
    # context_data = {
    # "task_context": current_state["task_description"],
    # "stress": current_state["stress"],
    # "trust": current_state["trust"],
    # "cognitive_load": current_state["cognitive_load"],
    # "game_score": current_state["game_score"],
    # "num_collisions": current_state["num_collisions"],
    # }
    current_state_df = pd.DataFrame([current_state])

    context_df = pd.concat([current_state_df, recent_context_df], ignore_index=True)
    print(context_df)
    return context_df


# Current user and game state
current_state = get_context_features(system_state)
recent_context_df = get_recent_context(current_state)
context_df = get_context_df(current_state, recent_context_df)


print("CONTEXT:----->\n",context_df)
# Generate adaptive explanation using RAG-based context window
response = Synthesizer.generate_response(question=current_state, context=context_df)

# Print response and thought process
print(f"ANSWER: \n{response.answer}")

print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print("\XAI Features Used:")
for feature in response.features:
    print(f"- {feature}")
    
print(f"\nContext: {response.enough_context}")

""" 
# Extract context
context_features = extract_system_context(system_state)

# Dynamically infer task description
task_description = infer_task_description(context_features)

# Current user and game state
current_state = {
    "stress": context_features["stress"],
    "trust": context_features["trust"],
    "cognitive_load": context_features["cognitive_load"],
    "game_score": context_features["system_performance"],  # Using performance as score
    "num_collisions": system_state.get("num_collisions", 0),
    "task_description": task_description,  # Auto-inferred task description
}

# Retrieve **recent interactions** (limit to last n period)
time_range = (datetime.now() - timedelta(days=20), datetime.now()) # make 10 secs

# ✅ Fix: Provide query_text for similarity search
recent_context = vec.search(
    query_text=current_state["task_description"],  # ✅ Required argument
    limit=3,
    # predicates=predicates,
    time_range=time_range
)
# Convert `recent_context` into a DataFrame if it's a list of dicts
recent_context_df = pd.DataFrame(recent_context)


# Convert `current_state` into a DataFrame with the same structure as `recent_context`
# current_state_df = pd.DataFrame([{
#     "id": "current_state",  # Placeholder ID to differentiate from DB records
#     "content": current_state["task_description"],
#     "embedding": None,  # No embedding needed for current state
#     "distance": 0.0,  # Distance is 0 since it's the direct current context
#     "trust": current_state["trust"],
#     "stress": current_state["stress"],
#     "created_at": pd.Timestamp.now().isoformat(),  # Current timestamp
#     "game_score": current_state["game_score"],
#     "cognitive_load": current_state["cognitive_load"],
#     "num_collisions": current_state["num_collisions"]
# }])
context_data = {
    "task_context": current_state["task_description"],
    "stress": current_state["stress"],
    "trust": current_state["trust"],
    "cognitive_load": current_state["cognitive_load"],
    "game_score": current_state["game_score"],
    "num_collisions": current_state["num_collisions"],
}
current_state_df = pd.DataFrame([current_state])

context_df = pd.concat([current_state_df, recent_context_df], ignore_index=True)
print(context_df)
# Generate adaptive explanation using RAG-based context window
response = Synthesizer.generate_response(question=current_state, context=context_df)

# Print response and thought process
print(f"\n{response.answer}")

print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print("\XAI Features Used:")
for feature in response.features:
    print(f"- {feature}")
    
print(f"\nContext: {response.enough_context}")
 """


# --------------------------------------------------------------
# Adaptive Explanation Retrieval (AdaX) - Context-Driven
# --------------------------------------------------------------

# Sample 1:
# Simulated current user and game state
current_state = {
    "stress": "high",
    "trust": "low",
    "cognitive_load": "high",
    "game_score": 65,
    "num_collisions": 5,
    "task_description": "Understanding why Overcooked AI chooses a specific path",
}


# Retrieve **recent interactions** (limit to last n period)
time_range = (datetime.now() - timedelta(days=10), datetime.now()) # make 10 secs

# ✅ Fix: Provide query_text for similarity search
recent_context = vec.search(
    query_text=current_state["task_description"],  # ✅ Required argument
    limit=3,
    # predicates=predicates,
    time_range=time_range
)
# Convert `recent_context` into a DataFrame if it's a list of dicts
recent_context_df = pd.DataFrame(recent_context)


# Convert `current_state` into a DataFrame with the same structure as `recent_context`
current_state_df = pd.DataFrame([{
    "id": "current_state",  # Placeholder ID to differentiate from DB records
    "content": current_state["task_description"],
    "embedding": None,  # No embedding needed for current state
    "distance": 0.0,  # Distance is 0 since it's the direct current context
    "trust": current_state["trust"],
    "stress": current_state["stress"],
    "created_at": pd.Timestamp.now().isoformat(),  # Current timestamp
    "game_score": current_state["game_score"],
    "cognitive_load": current_state["cognitive_load"],
    "num_collisions": current_state["num_collisions"]
}])

context_df = pd.concat([current_state_df, recent_context_df], ignore_index=True)
print(context_df)
# Generate adaptive explanation using RAG-based context window
response = Synthesizer.generate_response(question=current_state, context=context_df)
print(response)
# Print response and thought process
print(f"\n{response.answer}")

print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print("\XAI Features Used:")
for feature in response.features:
    print(f"- {feature}")
    
print(f"\nContext: {response.enough_context}")


# Sample 2:
# Simulated current user and game state
current_state = {
    "stress": "low",
    "trust": "low",
    "cognitive_load": "high",
    "game_score": 65,
    "num_collisions": 5,
    "task_description": "Understanding why AI chef picked up an onion",
}

# Retrieve **recent interactions** (limit to last n period)
time_range = (datetime.now() - timedelta(days=10), datetime.now()) # make 10 secs

# ✅ Fix: Provide query_text for similarity search
recent_context = vec.search(
    query_text=current_state["task_description"],  # ✅ Required argument
    limit=3,
    # predicates=predicates,
    time_range=time_range
)
# Convert `recent_context` into a DataFrame if it's a list of dicts
recent_context_df = pd.DataFrame(recent_context)


# Convert `current_state` into a DataFrame with the same structure as `recent_context`
current_state_df = pd.DataFrame([{
    "id": "current_state",  # Placeholder ID to differentiate from DB records
    "content": current_state["task_description"],
    "embedding": None,  # No embedding needed for current state
    "distance": 0.0,  # Distance is 0 since it's the direct current context
    "trust": current_state["trust"],
    "stress": current_state["stress"],
    "created_at": pd.Timestamp.now().isoformat(),  # Current timestamp
    "game_score": current_state["game_score"],
    "cognitive_load": current_state["cognitive_load"],
    "num_collisions": current_state["num_collisions"]
}])

context_df = pd.concat([current_state_df, recent_context_df], ignore_index=True)
print(context_df)
# Generate adaptive explanation using RAG-based context window
response = Synthesizer.generate_response(question=current_state, context=context_df)

# Print response and thought process
print(f"\n{response.answer}")

print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print("\XAI Features Used:")
for feature in response.features:
    print(f"- {feature}")
    
print(f"\nContext: {response.enough_context}")
