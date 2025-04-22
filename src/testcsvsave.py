# import csv
# import json
# import os
# from datetime import datetime

# # Setup directory and CSV path
# csv_dir = 'metrics_logs'
# os.makedirs(csv_dir, exist_ok=True)
# csv_path = os.path.join(csv_dir, 'metrics_log.csv')

# # Define the CSV headers
# CSV_FIELDS = ["timestamp", "player_id", "round_id", "uid", "type", "data"]

# # Write header if file doesn't exist
# if not os.path.exists(csv_path):
#     with open(csv_path, mode='w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
#         writer.writeheader()

# # === Function to write a sample_obj ===
# def write_sample(sample_obj):
#     row = {
#         "timestamp": sample_obj["timestamp"],
#         "player_id": sample_obj["player_id"],
#         "round_id": sample_obj["round_id"],
#         "uid": sample_obj["uid"],
#         "type": sample_obj["type"],
#         "data": json.dumps(sample_obj["data"])  # <-- this keeps the dict as a single string
#     }
#     with open(csv_path, mode='a', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
#         writer.writerow(row)

# # === Example Data Buffers ===

# # Example performance metrics buffer
# met_labels = ['eng.isActive', 'eng', 'exc.isActive', 'exc', 'lex', 'str.isActive', 'str', 'rel.isActive', 'rel', 'int.isActive', 'int', 'foc.isActive', 'foc']
# met_buffer = [
#     (datetime.now().isoformat(), [True, 0.7, True, 0.9, 0.3, False, 0.4, True, 0.6, True, 0.5, False, 0.8])
# ]

# # Metadata
# player_id = "P01"
# round_id = 1
# uid = "abc123"

# # Write MET sample
# for timestamp, sample in met_buffer:
#     sample_obj = {
#         "timestamp": timestamp,
#         "player_id": player_id,
#         "round_id": round_id,
#         "uid": uid,
#         "type": "met",
#         "data": dict(zip(met_labels, sample))
#     }
#     write_sample(sample_obj)

# # Write model prediction
# model_output = {
#     "prediction": "excitement",
#     "probabilities": {
#         "excitement": 0.9462,
#         "stress": 0.00005,
#         "depression": 0.00071,
#         "relaxation": 0.0529
#     }
# }
# sample_obj = {
#     "timestamp": datetime.now().isoformat(),
#     "player_id": player_id,
#     "round_id": round_id,
#     "uid": uid,
#     "type": "model",
#     "data": model_output
# }
# write_sample(sample_obj)

import json


def rule_based_static_explanation(game_data):
    """
    Generate a static explanation based on AI's current state using simple rules.
    This is tailored for an onion-only Overcooked game.
    """
    try:
        parsed_state = json.loads(game_data.get("state", "{}"))
        players = parsed_state.get("players", [])
        objects = parsed_state.get("objects", [])

        ai_index = 1 if game_data.get("player_0_is_human", True) else 0
        ai_player = players[ai_index] if len(players) > ai_index else {}
        held_object = ai_player.get("held_object")

        explanation_parts = []

        # Rule 1: AI is holding onion
        if held_object and held_object.get("name") == "onion":
            explanation_parts.append("I picked an onion to cook soup.")

        # Rule 2: AI is holding dish
        elif held_object and held_object.get("name") == "dish":
            explanation_parts.append("I grabbed a dish to plate the soup.")

        # Rule 3: AI is holding soup
        elif held_object and held_object.get("name") == "soup":
            if held_object.get("is_ready"):
                explanation_parts.append("I picked up the ready soup to deliver.")
            else:
                explanation_parts.append("I picked soup, waiting for it to be ready.")

        # Rule 4: Soup object analysis
        for obj in objects:
            if obj.get("name") == "soup":
                ingredients = obj.get("_ingredients", [])
                is_cooking = obj.get("is_cooking", False)
                is_ready = obj.get("is_ready", False)

                if is_ready:
                    explanation_parts.append("Soup is ready. I’ll deliver it now.")
                elif is_cooking:
                    tick = obj.get("cooking_tick", 0)
                    explanation_parts.append(f"Soup is cooking. {20 - tick} timesteps left.")
                elif len(ingredients) < 3:
                    explanation_parts.append("Soup pot needs more onions.")
                else:
                    explanation_parts.append("Soup pot is full, starting to cook.")
                break

        # Fallback rule if no other matches
        if not explanation_parts:
            explanation_parts.append("I’m repositioning to prepare for the next step.")

        return " ".join(explanation_parts)

    except Exception as e:
        return f"Static explanation error: {str(e)}"


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
                # if len(obj.get("_ingredients", [])) < 3:
                #     return "Pot needs more onions."

        return "I’m moving to prepare next task."

    except Exception as e:
        return f"Standard static explanation error: {str(e)}"


# Example usage
if __name__ == "__main__":
    game_data = {'state': '{"players": [{"position": [5, 2], "orientation": [-1, 0], "held_object": {"name": "dish", "position": [5, 2]}}, {"position": [1, 3], "orientation": [0, 1], "held_object": {"name": "dish", "position": [1, 3]}}], "objects": [{"name": "soup", "position": [4, 2], "_ingredients": [{"name": "onion", "position": [4, 2]}, {"name": "onion", "position": [4, 2]}, {"name": "onion", "position": [4, 2]}], "cooking_tick": 13, "is_cooking": true, "is_ready": false, "is_idle": false, "cook_time": 20, "_cooking_tick": 13}], "bonus_orders": [], "all_orders": [{"ingredients": ["onion", "onion", "onion"]}], "timestep": 38}', 'joint_action': '[[0, 0], [-1, 0]]', 'reward': 0, 'time_left': 3.3504509925842285, 'score': 0, 'time_elapsed': 16.64954900741577, 'cur_gameloop': 39, 'layout': '[["X", "X", "X", "X", "X", "X", "X", "X", "X"], ["O", " ", "X", "S", "X", "O", "X", " ", "S"], ["X", " ", " ", " ", "P", " ", " ", " ", "X"], ["X", " ", " ", " ", "P", " ", " ", " ", "X"], ["X", "X", "X", "D", "X", "D", "X", "X", "X"]]', 'layout_name': 'asymmetric_advantages', 'trial_id': '1744636630.231665', 'player_0_id': 'e1fb292721a24f2c9d13e03169644c95', 'player_1_id': 'PPOCoordinationRing_1', 'player_0_is_human': True, 'player_1_is_human': False, 'collision': False, 'num_collisions': 0, 'unix_timestamp': 1744636646.881214, 'playerId': 'e1fb292721a24f2c9d13e03169644c95'}

    print(rule_based_static_explanation(game_data))
