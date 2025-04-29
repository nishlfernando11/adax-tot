from datetime import datetime, timedelta
from app.database.vector_store import VectorStore
from app.services.synthesizer_mistral import Synthesizer
from timescale_vector import client
import pandas as pd
import json
import math
# Initialize VectorStore
# vec = VectorStore()


def extract_system_context(state):
    """
    Extracts key user-AI interaction elements from any system.

    Parameters:
    - system_state (dict): AI system state including user metrics & AI decisions.

    Returns:
    - (dict): Extracted general features applicable across domains.
    """
    print("Extracting system context...", state)
    # return {
    #     # "ai_action": system_state.get("ai_action", "unknown"),
    #     # "user_action": system_state.get("user_action", "unknown"),
    #     # "ai_decision_confidence": system_state.get("ai_decision_confidence", "="),
    #     # "system_performance": system_state.get("system_performance", 0),
    #     # "risk_perception": system_state.get("risk_perception", ""),
    #     "stress": state["physiological_state"]["stress"],
    #     # "trust": state["physiological_state"]["trust"], 
    #     "cognitive_load": state["physiological_state"]["cognitive_load"],
    #     "num_collisions": state["behavioral_state"]["num_collisions"],
    #     "score": state["behavioral_state"]["score"],
    #     "time_left": state["behavioral_state"]["time_left"]
    # }
    
    return {
    "stress": (state.get("physiological_state") or {}).get("stress"),
    # "trust": (state.get("physiological_state") or {}).get("trust"),
    "cognitive_load": (state.get("physiological_state") or {}).get("cognitive_load"),
    "num_collisions": (state.get("behavioral_state") or {}).get("num_collisions"),
    "score": (state.get("behavioral_state") or {}).get("score"),
    "time_left": (state.get("behavioral_state") or {}).get("time_left"),
    "playerId": (state.get("behavioral_state") or {}).get("playerId"),
    "layout_name": (state.get("behavioral_state") or {}).get("layout_name")
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



# Mapping from orientation vectors to direction labels
ORIENTATION_TO_DIRECTION = {
    (0, -1): "UP",
    (0, 1): "DOWN",
    (1, 0): "RIGHT",
    (-1, 0): "LEFT",
    (0, 0): "STAY"
}

# Action mapping (for reference)
ACTIONS = {
    "STAY": (0, 0),
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "SPACE": "interact"
}


# def summarize_player_status(data_obj):
#     try:
#         behavioral = data_obj.get("behavioral_state", {})
#         game_state_raw = behavioral.get("state", '{}')
#         game_state = json.loads(game_state_raw)
#         timestep = game_state.get("timestep", '?')
#         layout_name = behavioral.get("layout_name", '?')

#         players = game_state.get("players", [])
#         p0_is_human = behavioral.get("player_0_is_human", False)
#         p1_is_human = behavioral.get("player_1_is_human", False)

#         player_info = []
#         for idx, player in enumerate(players):
#             role = "human" if (p0_is_human if idx == 0 else p1_is_human) else "AI"
#             position = player.get("position", '?')
#             orientation_tuple = tuple(player.get("orientation", (0, 0)))
#             orientation = ORIENTATION_TO_DIRECTION.get(orientation_tuple, str(orientation_tuple))
#             held_object = player.get("held_object")

#             if held_object is None:
#                 obj_status = "holding nothing"
#             else:
#                 obj_name = held_object.get("name", "an object")
#                 obj_ready = held_object.get("is_ready", False)
#                 obj_status = f"holding {obj_name}{' (ready)' if obj_ready else ''}"

#             player_info.append(f"The {role} player is at {position}, facing {orientation}, and is {obj_status}.")

#         # Layout and performance metrics
#         score = behavioral.get("score", 0)
#         time_elapsed = behavioral.get("time_elapsed", 1e-6)  # Avoid div by 0
#         num_collisions = behavioral.get("num_collisions", 0)
#         collision_flag = behavioral.get("collision", False)

#         score_rate = round(score / time_elapsed, 2)
#         collision_rate = round(num_collisions / time_elapsed, 2)

#         performance_summary = f"Layout: {layout_name}. Score rate: {score_rate}/s. Collision rate: {collision_rate}/s."
#         if collision_flag:
#             performance_summary += " A collision just occurred."

#         return f"At timestep {timestep}, {performance_summary} " + " ".join(player_info)

#     except Exception as e:
#         return f"Player status summary error: {str(e)}"

def summarize_player_status(data_obj):
    try:
        behavioral = data_obj.get("behavioral_state", {})
        game_state_raw = behavioral.get("state", '{}')
        game_state = json.loads(game_state_raw)
        timestep = game_state.get("timestep", '?')
        layout_name = behavioral.get("layout_name", '?')

        players = game_state.get("players", [])
        p0_is_human = behavioral.get("player_0_is_human", False)
        p1_is_human = behavioral.get("player_1_is_human", False)

        player_positions = []
        player_info = []
        for idx, player in enumerate(players):
            role = "human" if (p0_is_human if idx == 0 else p1_is_human) else "AI"
            position = player.get("position", '?')
            player_positions.append(position)
            orientation_tuple = tuple(player.get("orientation", (0, 0)))
            orientation = ORIENTATION_TO_DIRECTION.get(orientation_tuple, str(orientation_tuple))
            held_object = player.get("held_object")

            if held_object is None:
                obj_status = "holding nothing"
            else:
                obj_name = held_object.get("name", "an object")
                obj_ready = held_object.get("is_ready", False)
                obj_status = f"holding {obj_name}{' (ready)' if obj_ready else ''}"

            player_info.append(f"The {role} player is at {position}, facing {orientation}, and is {obj_status}.")

        # Layout and performance metrics
        score = behavioral.get("score", 0)
        time_elapsed = behavioral.get("time_elapsed", 1e-6)  # Avoid div by 0
        num_collisions = behavioral.get("num_collisions", 0)
        collision_flag = behavioral.get("collision", False)
        time_left = behavioral.get("time_left", 0)

        score_rate = round(score / time_elapsed, 2)
        collision_rate = round(num_collisions / time_elapsed, 2)

        # Ingredient progress (from soups in objects)
        objects = game_state.get("objects", [])
        completed_ingredients = 0
        for obj in objects:
            if obj.get("name") == "soup":
                completed_ingredients += len(obj.get("_ingredients", []))

        total_required_ingredients = 0
        orders = game_state.get("all_orders", [])
        for order in orders:
            total_required_ingredients += len(order.get("ingredients", []))

        ingredient_progress = f"Ingredients delivered: {completed_ingredients}/{total_required_ingredients}."

        # Teammate proximity (Euclidean distance)
        if len(player_positions) == 2:
            dx = player_positions[0][0] - player_positions[1][0]
            dy = player_positions[0][1] - player_positions[1][1]
            dist = math.sqrt(dx**2 + dy**2)
            proximity = f"Player distance: {round(dist, 2)} tiles."
        else:
            proximity = "Player proximity unknown."

        # Time urgency
        time_tag = "Low time left!" if time_left < 10 else "Time remaining is adequate."

        performance_summary = (
            f"Layout: {layout_name}. Score rate: {score_rate}/s. Collision rate: {collision_rate}/s. "
            f"{ingredient_progress} {proximity} {time_tag}"
        )

        if collision_flag:
            performance_summary += " A collision just occurred."

        return f"At timestep {timestep}, {performance_summary} " + " ".join(player_info)

    except Exception as e:
        return f"Player status summary error: {str(e)}"


import json
import math

ORIENTATION_TO_DIRECTION = {
    (0, -1): "up",
    (0, 1): "down",
    (-1, 0): "left",
    (1, 0): "right",
    (0, 0): "still"
}

def summarize_full_context_detailed(data_obj, adax_data=None, physio_inference=None):
    try:
        behavioral = data_obj.get("behavioral_state", {})
        game_state_raw = behavioral.get("state", '{}')
        game_state = json.loads(game_state_raw)
        timestep = game_state.get("timestep", '?')
        layout_name = behavioral.get("layout_name", '?')

        players = game_state.get("players", [])
        p0_is_human = behavioral.get("player_0_is_human", False)
        p1_is_human = behavioral.get("player_1_is_human", False)

        player_positions = []
        player_info = []
        for idx, player in enumerate(players):
            role = "human" if (p0_is_human if idx == 0 else p1_is_human) else "AI"
            position = player.get("position", '?')
            player_positions.append(position)
            orientation_tuple = tuple(player.get("orientation", (0, 0)))
            orientation = ORIENTATION_TO_DIRECTION.get(orientation_tuple, str(orientation_tuple))
            held_object = player.get("held_object")

            if held_object is None:
                obj_status = "holding nothing"
            else:
                obj_name = held_object.get("name", "an object")
                obj_ready = held_object.get("is_ready", False)
                obj_status = f"holding {obj_name}{' (ready)' if obj_ready else ''}"

            player_info.append(f"The {role} player is at {position}, facing {orientation}, and is {obj_status}.")

        score = behavioral.get("score", 0)
        time_elapsed = behavioral.get("time_elapsed", 1e-6)
        num_collisions = behavioral.get("num_collisions", 0)
        collision_flag = behavioral.get("collision", False)
        time_left = behavioral.get("time_left", 0)

        score_rate = round(score / time_elapsed, 2)
        collision_rate = round(num_collisions / time_elapsed, 2)

        objects = game_state.get("objects", [])
        soup_summaries = []
        for obj in objects:
            if obj.get("name") == "soup":
                ing_count = len(obj.get("_ingredients", []))
                is_cooking = obj.get("is_cooking", False)
                is_ready = obj.get("is_ready", False)
                tick = obj.get("cooking_tick", 0)

                if is_ready:
                    soup_summaries.append("One soup is ready.")
                elif is_cooking:
                    soup_summaries.append(f"Soup cooking, {20 - tick} timesteps left.")
                elif ing_count < 3:
                    soup_summaries.append(f"Pot has {ing_count}/3 onions.")
                else:
                    soup_summaries.append("Pot is full, ready to cook.")

        soup_status = " ".join(soup_summaries) if soup_summaries else "No soups being prepared."

        orders = game_state.get("all_orders", [])
        total_required_ingredients = sum(len(order.get("ingredients", [])) for order in orders)
        completed_ingredients = sum(len(obj.get("_ingredients", [])) for obj in objects if obj.get("name") == "soup")

        ingredient_progress = f"Ingredients (onions) in pots: {completed_ingredients} of {total_required_ingredients}."
        completed_orders = f"Completed orders: {int(score / 20)}."

        if len(player_positions) == 2:
            dx = player_positions[0][0] - player_positions[1][0]
            dy = player_positions[0][1] - player_positions[1][1]
            dist = math.sqrt(dx ** 2 + dy ** 2)
            proximity = f"Player distance: {round(dist, 2)} tiles."
        else:
            proximity = "Player proximity unknown."

        time_tag = "Low time left!" if time_left < 10 else "Time remaining is adequate."

        performance_summary = (
            f"Layout: {layout_name}, timestep: {timestep}. Score rate: {score_rate}/s, Collision rate: {collision_rate}/s.\n"
            f"{ingredient_progress} {soup_status} {proximity} {time_tag} (time left {time_left}) {completed_orders}"
        )

        if collision_flag:
            performance_summary += " A collision just occurred."

        if physio_inference:
            stress = physio_inference.get("stress", '?')
            trust = physio_inference.get("trust", '?')
            cogload = physio_inference.get("cognitive_load", '?')
            performance_summary += f" | User state: stress={stress}, trust={trust}, cognitive_load={cogload}."

        if adax_data:
            features = adax_data.get("features", [])
            performance_summary += f" | Explanation features: {', '.join(features)}."

        return f"{performance_summary} {' '.join(player_info)}"

    except Exception as e:
        return f"Full context summary error: {str(e)}"

def summarize_full_context(data_obj, adax_data=None, physio_inference=None):
    print(data_obj)
    try:
        behavioral = data_obj.get("behavioral_state", {})
        game_state_raw = behavioral.get("state", '{}')
        game_state = json.loads(game_state_raw)
        timestep = game_state.get("timestep", '?')
        layout_name = behavioral.get("layout_name", '?')

        players = game_state.get("players", [])
        p0_is_human = behavioral.get("player_0_is_human", False)
        p1_is_human = behavioral.get("player_1_is_human", False)

        player_positions = []
        player_info = []
        for idx, player in enumerate(players):
            role = "human" if (p0_is_human if idx == 0 else p1_is_human) else "AI"
            position = player.get("position", '?')
            player_positions.append(position)
            orientation_tuple = tuple(player.get("orientation", (0, 0)))
            orientation = ORIENTATION_TO_DIRECTION.get(orientation_tuple, str(orientation_tuple))
            held_object = player.get("held_object")

            if held_object is None:
                obj_status = "holding nothing"
            else:
                obj_name = held_object.get("name", "an object")
                obj_ready = held_object.get("is_ready", False)
                obj_status = f"holding {obj_name}{' (ready)' if obj_ready else ''}"

            player_info.append(f"The {role} player is at {position}, facing {orientation}, and is {obj_status}.")

        score = behavioral.get("score", 0)
        time_elapsed = behavioral.get("time_elapsed", 1e-6)
        num_collisions = behavioral.get("num_collisions", 0)
        collision_flag = behavioral.get("collision", False)
        time_left = behavioral.get("time_left", 0)

        score_rate = round(score / time_elapsed, 2)
        collision_rate = round(num_collisions / time_elapsed, 2)

        objects = game_state.get("objects", [])
        completed_ingredients = sum(len(obj.get("_ingredients", [])) for obj in objects if obj.get("name") == "soup")

        orders = game_state.get("all_orders", [])
        total_required_ingredients = sum(len(order.get("ingredients", [])) for order in orders)

        ingredient_progress = f"Ingredients (onions) delivered for current order: {completed_ingredients} of {total_required_ingredients}."
        
        completed_orders = int(score/20)
        completed_orders = f"Completed Orders is " + str(completed_orders) + "."

        if len(player_positions) == 2:
            dx = player_positions[0][0] - player_positions[1][0]
            dy = player_positions[0][1] - player_positions[1][1]
            dist = math.sqrt(dx**2 + dy**2)
            proximity = f"Player distance: {round(dist, 2)} tiles."
        else:
            proximity = "Player proximity unknown."

        time_tag = "Low time left!" if time_left < 10 else "Time remaining is adequate."

        performance_summary = (
            f"Layout: {layout_name}, timestep: {timestep}. Score rate: {score_rate}/s, Collision rate: {collision_rate}/s, Score: {score_rate}, Collisions: {num_collisions}. "
            f"{ingredient_progress} {proximity} {time_tag} {completed_orders}"
        )
        if collision_flag:
            performance_summary += " A collision just occurred."

        # Add physiological inference summary
        if physio_inference:
            stress = physio_inference.get("stress", '?')
            trust = physio_inference.get("trust", '?')
            cogload = physio_inference.get("cognitive_load", '?')
            performance_summary += f" | User state is stress {stress}, trust {trust}, cognitive load {cogload}."

        # Add AdaX features if available
        if adax_data:
            features = adax_data.get("features", [])
            performance_summary += f" | Explanation features: {', '.join(features)}."

        return f"{performance_summary} {' '.join(player_info)}"

    except Exception as e:
        return f"Full context summary error: {str(e)}"


def standard_rule_based_explanation(game_data):
    """
    Simpler and more standard static explanation logic for onion-only Overcooked gameplay.
    Prioritizes minimal and non-overlapping decision rules.
    """
    try:
        parsed_state = json.loads(game_data.get("state", "{}"))
        players = parsed_state.get("players", [])
        objects = parsed_state.get("objects", [])

        ai_index = 1 if game_data.get("player_0_is_human", True) else 0
        ai_player = players[ai_index] if len(players) > ai_index else {}
        held_object = ai_player.get("held_object")

        if held_object:
            obj_name = held_object.get("name")
            if obj_name == "onion":
                return "I picked onion to prepare soup."
            if obj_name == "dish":
                return "I picked dish to serve soup."
            if obj_name == "soup":
                return "I picked soup to deliver."

        for obj in objects:
            if obj.get("name") == "soup":
                if obj.get("is_ready"):
                    return "Soup is ready to be delivered."
                if obj.get("is_cooking"):
                    return "Soup is cooking."
                if len(obj.get("_ingredients", [])) < 3:
                    return "Pot needs more onions."

        return "I’m going for soup preparation."

    except Exception as e:
        return f"Standard static explanation error: {str(e)}"


def get_context_features(state):
    # Extract context
    context_features = extract_system_context(state)

    # Dynamically infer task description
    task_description = infer_task_description(context_features)

    # Current user and game state
    current_state = {
        "playerId": context_features["playerId"],
        "layout_name": context_features["layout_name"],
        "stress": context_features["stress"],
        # "trust": context_features["trust"],
        "cognitive_load": context_features["cognitive_load"],
        "score": context_features["score"],
        "num_collisions": context_features["num_collisions"],
        "time_left": context_features["time_left"],
        "task_description": task_description,  # Manual static task description
    }
    
    return current_state

def get_recent_context(current_state, vector_db, time_unit="days", time_value=30, return_dataframe=True):
    # Retrieve **recent interactions** (limit to last n period)
    # @TODO: Replace with actual time range
    # TODO:make 10 secs, or accept time range as input
    if time_unit == "days":
        time_range = (datetime.now() - timedelta(days=time_value), datetime.now())
    elif time_unit == "seconds":
        time_range = (datetime.now() - timedelta(seconds=time_value), datetime.now())
    elif time_unit == "minutes":
        time_range = (datetime.now() - timedelta(minutes=time_value), datetime.now())
    elif time_unit == "hours":
        time_range = (datetime.now() - timedelta(hours=time_value), datetime.now())
    else:
        raise ValueError("Invalid time unit. Use 'days', 'seconds', 'minutes', or 'hours'.")
    query = format_content(current_state)

    print("$$$$------layout name", current_state["layout_name"])
    # ✅ Fix: Provide query_text for similarity search
    recent_context = vector_db.search(
        query_text= query, #get by playerId
        limit=3,
        predicates=None,
        time_range=time_range,
        return_dataframe=return_dataframe
    )
    print("===========>recent_context", recent_context)
    
    # Convert `recent_context` into a DataFrame if it's a list of dicts
    recent_context_df = pd.DataFrame(recent_context.loc[:, recent_context.columns != 'embedding'])
    print("===========>recent_context_df", recent_context_df)
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
    # content = (
    #     f"Game state: Score {state['score']}, {state['num_collisions']} collisions,\n"
    #     f"{state['time_left']} seconds left.\n"
    #     f"User state: Stress {state['stress']}, cognitive load {state['cognitive_load']}.\n"
    #     # f"Task: {state['task_description']}\n"
    #     # f"Best Explanation: {state['explanation_simplified']}\n"
    #     # f"Alternative Explanation (Balanced): {state['explanation_balanced']}\n"
    #     # f"Step-by-Step Explanation: {state['explanation_step_by_step']}\n"
    #     # f"duration: {state.get('explanation_duration', 'medium')},  # Default: 'medium'\n"
    #     # f"granularity: {state.get('explanation_granularity', 'steps')},  # Default: 'steps'\n"
    #     # f"timing: {state.get('explanation_timing', 'reactive')}  # Default: 'reactive'"
    # )
    
    # get by playerId
    print("----> ", state)
    content = (
        f"playerId: {state.get('playerId')}\n"
        f"layout_name: {state.get('layout_name')}\n"
    )
    return content
