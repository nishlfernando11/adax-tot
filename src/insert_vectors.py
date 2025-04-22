from datetime import datetime
import os

import pandas as pd
from app.database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from app.config.settings import EMBEDDING_SIZE
from functools import partial

def format_content(row):
    content = (
        f"Player ID: {row['playerId']}\n"
        f"Kitchen layout: {row['layout_name']}\n"
        f"Game state: Score {row['score']}, {row['num_collisions']} collisions. Score rate: {row['score_rate']}, Collision rate: {row['collision_rate']} \n"
        f"{row['time_left']} seconds left.\n"
        f"User state: Stress {row['stress']}, trust {row['trust']}, cognitive load {row['cognitive_load']}.\n"
        # f"Task: {row['task_description']}\n"
        f"Final Explanation: {row['final_explanation']}\n"
        # f"Best Explanation: {row['explanation_simplified']}\n"
        # f"Alternative Explanation (Balanced): {row['explanation_balanced']}\n"
        # f"Step-by-Step Explanation: {row['explanation_step_by_step']}\n"
        f"Justification: {row['justification']}\n"
        f"duration: {row.get('explanation_duration', 'medium')},  # Default: 'medium'\n"
        f"granularity: {row.get('explanation_granularity', 'steps')},  # Default: 'steps'\n"
        f"timing: {row.get('explanation_timing', 'reactive')}  # Default: 'reactive'"
    )
    return content

def extract_keywords(explanation: str):
    import re
    verbs = ["picked", "dropped", "moved", "placed", "delivered", "grabbed"]
    objects = ["onion", "dish", "soup", "pot", "station"]
    keywords = []
    for verb in verbs:
        for obj in objects:
            if f"{verb} {obj}" in explanation:
                keywords.append(verb)
                keywords.append(obj)
    # Fallback: split important words
    if not keywords:
        keywords = re.findall(r'\b(onion|dish|pot|soup|deliver|pick|drop|plate)\b', explanation)
    return list(set(keywords))


def process_row(row, vec):
    from datetime import datetime
    import uuid
    import numpy as np

    # print("Processing row:", row)  # Debugging step

    # --- Validate Explanation: Does AI mention a valid action? ---
    ai_summary = row.get("ai_action_summary", "").lower()
    final_expl = row.get("final_explanation", "").lower()
    ai_action_ok = all(part in ai_summary for part in extract_keywords(final_expl))

    # --- Format pot status summary ---
    pot_summary = [
        f"pot at {p['position']}, onions={p['onions']}, cooking={p['is_cooking']}"
        for p in row.get("pot_status", [])
    ]

    # --- Format content for embedding ---
    contents = format_content(row)

    # --- Generate embedding ---
    embedding = vec.get_embedding_ollama(contents)
    print("Embedding shape:", np.array(embedding).shape)
    assert len(embedding) == EMBEDDING_SIZE, f"Embedding size mismatch! Expected {EMBEDDING_SIZE}, got {len(embedding)}"

    return pd.Series({
        "id": str(uuid_from_time(datetime.now())),
        "metadata": {
            "timestep": row.get("timestep", -1),
            "playerId": row["playerId"],
            "stress": row["stress"],
            "trust": row["trust"],
            "cognitive_load": row["cognitive_load"],
            "layout_name": row["layout_name"],
            "score": row["score"],
            "score_rate": row["score_rate"],
            "num_collisions": row["num_collisions"],
            "collision_rate": row["collision_rate"],
            "time_left": row["time_left"],
            "final_explanation": row["final_explanation"],
            "justification": row["justification"],
            "explanation_features": {
                "explanation_duration": row["explanation_duration"],
                "explanation_granularity": row["explanation_granularity"],
                "explanation_timing": row["explanation_timing"],
            },
            "player_actions": row["player_actions"],
            "held_objects": row["held_objects"],
            "ai_action_summary": row["ai_action_summary"],
            "pot_status": pot_summary,
            "valid_explanation_actions": ai_action_ok,
            "created_at": datetime.now().isoformat(),
        },
        "contents": contents,
        "embedding": embedding,
    })


def process_row_old(row, vec):
    print("<-------->Processing row:", row)  # Debugging step
    contents = format_content(row)
    embedding = vec.get_embedding_ollama(contents)

    print("Embedding shape:", np.array(embedding).shape)  # Debugging step
    assert len(embedding) == EMBEDDING_SIZE, f"Embedding size mismatch! Expected {EMBEDDING_SIZE}, got {len(embedding)}"
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "timestep": row["timestep"],
                "playerId": row["playerId"],
                "stress": row["stress"],
                "trust": row["trust"],
                "cognitive_load": row["cognitive_load"],
                "layout_name" : row["layout_name"],
                "score": row["score"],
                "score_rate": row["score_rate"],
                "num_collisions": row["num_collisions"],
                "collision_rate": row["collision_rate"],
                "time_left": row["time_left"],
                "final_explanation": row["final_explanation"],
                "justification": row["justification"],
                "explanation_features": {"explanation_duration":row["explanation_duration"], "explanation_granularity":row["explanation_granularity"], "explanation_timing":row["explanation_timing"]},
                "created_at": datetime.now().isoformat(),
            },
            "contents": contents,
            "embedding": embedding,
        }
    )


def run():
    
    # Initialize VectorStore
    vec = VectorStore()

    # Read the CSV file
    file_path = os.path.abspath("data/userdata.csv")  # Move up 2 levels

    print("Loading file from:", file_path)

    print("Reading AdaX dataset...")
    df = pd.read_csv(file_path, sep=";")
    print("AdaX dataset read successfully.\n", df)
    print("Preparing records for insertion...")
    global process_row
    process_row  = partial(process_row, vec=vec)
    # Create a new DataFrame with the processed records
    records_df = df.apply(process_row, axis=1)
    
    ## uncomment below on first time run
    # vec.create_tables()
    # vec.create_index()  # DiskAnnIndex
    ## end comment
    
    vec.upsert(records_df)
    
## uncomment below on first time run
# run()
 ## end comment

# with ThreadPoolExecutor(max_workers=4) as executor:
#     embeddings = list(executor.map(process_row, df))


