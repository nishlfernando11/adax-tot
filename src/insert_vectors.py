from datetime import datetime
import os

import pandas as pd
from app.database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

# Initialize VectorStore
vec = VectorStore()

# Read the CSV file
# df = pd.read_csv("../data/faq_dataset.csv", sep=";")

file_path = os.path.abspath("data/userdata.csv")  # Move up 2 levels

print("Loading file from:", file_path)

print("Reading AdaX dataset...")
df = pd.read_csv(file_path, sep=";")
print("AdaX dataset read successfully.\n", df)
print("Preparing records for insertion...")


# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store.

    This function creates a record with a UUID version 1 as the ID, which captures
    the current time or a specified time.

    Note:
        - By default, this function uses the current time for the UUID.
        - To use a specific time:
          1. Import the datetime module.
          2. Create a datetime object for your desired time.
          3. Use uuid_from_time(your_datetime) instead of uuid_from_time(datetime.now()).

        Example:
            from datetime import datetime
            specific_time = datetime(2023, 1, 1, 12, 0, 0)
            id = str(uuid_from_time(specific_time))

        This is useful when your content already has an associated datetime.
    """
    # content = f"Question: {row['question']}\nAnswer: {row['answer']}"
    # embedding = vec.get_embedding(content)
    # return pd.Series(
    #     {
    #         "id": str(uuid_from_time(datetime.now())),
    #         "metadata": {
    #             "category": row["category"],
    #             "created_at": datetime.now().isoformat(),
    #         },
    #         "contents": content,
    #         "embedding": embedding,
    #     }
    # )
    
    """
    Generates an embedding for adaptive explanations and formats the data for vector storage.
    
    Parameters:
    row (pd.Series): A row from the adaptive explanations dataset.
    vec (EmbeddingModel): An embedding model instance with a `get_embedding` method.

    Returns:
    pd.Series: A new row with an embedding and metadata.
    """
    # # Construct content based on the adapted AdaX format
    # content = (
    #     f"Task: {row['task_description']}\n"
    #     f"User State: Stress={row['stress']}, Trust={row['trust']}, Cognitive Load={row['cognitive_load']}\n"
    #     f"Game Metrics: Score={row['score']}, Collisions={row['num_collisions']}\n"
    #     f"Explanation (Simplified): {row['explanation_simplified']}\n"
    #     f"Explanation (Balanced): {row['explanation_balanced']}\n"
    #     f"Explanation (Step-by-Step): {row['explanation_step_by_step']}"
    # )

    # Construct content using the adapted AdaX format (previously "question;answer;category")
    
    content = (
        f"Task: {row['task_description']}\n"
        f"Final Explanation: {row['final_explanation']}\n"
        # f"Best Explanation: {row['explanation_simplified']}\n"
        # f"Alternative Explanation (Balanced): {row['explanation_balanced']}\n"
        # f"Step-by-Step Explanation: {row['explanation_step_by_step']}"
    )
    # Generate embedding vector
    embedding =  vec.get_embedding_ollama(content)
    # embedding =  vec.get_embedding_openai(content)

    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "stress": row["stress"],
                "trust": row["trust"],
                "cognitive_load": row["cognitive_load"],
                "score": row["score"],
                "num_collisions": row["num_collisions"],
                "time_left": row["time_left"],
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )


# records_df = df.apply(prepare_record, axis=1)

# # Create tables and insert data
# vec.create_tables()
# vec.create_index()  # DiskAnnIndex
# vec.upsert(records_df)


from concurrent.futures import ThreadPoolExecutor
import numpy as np

def format_content(row):
    content = (
        f"Game state: Score {row['score']}, {row['num_collisions']} collisions,\n"
        f"{row['time_left']} seconds left.\n"
        f"User state: Stress {row['stress']}, trust {row['trust']}, cognitive load {row['cognitive_load']}.\n"
        f"Task: {row['task_description']}\n"
        f"Final Explanation: {row['final_explanation']}\n"
        # f"Best Explanation: {row['explanation_simplified']}\n"
        # f"Alternative Explanation (Balanced): {row['explanation_balanced']}\n"
        # f"Step-by-Step Explanation: {row['explanation_step_by_step']}\n"
        f"duration: {row.get('explanation_duration', 'medium')},  # Default: 'medium'\n"
        f"granularity: {row.get('explanation_granularity', 'steps')},  # Default: 'steps'\n"
        f"timing: {row.get('explanation_timing', 'reactive')}  # Default: 'reactive'"
    )
    return content

def process_row(row):
     
    # text_data = f"Game state: Score {row['score']}, {row['num_collisions']} collisions, {row['time_left']} seconds left. User state: Stress {row['stress']}, trust {row['trust']}, cognitive load {row['cognitive_load']}."
    contents = format_content(row)
    embedding = vec.get_embedding_ollama(contents)

    print("Embedding shape:", np.array(embedding).shape)  # Debugging step
    assert len(embedding) == 1024, f"Embedding size mismatch! Expected 1024, got {len(embedding)}"

    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "stress": row["stress"],
                "trust": row["trust"],
                "cognitive_load": row["cognitive_load"],
                "score": row["score"],
                "num_collisions": row["num_collisions"],
                "time_left": row["time_left"],
                "final_explanation": row["final_explanation"],
                "explanation_features": {"explanation_duration":row["explanation_duration"], "explanation_granularity":row["explanation_granularity"], "explanation_timing":row["explanation_timing"]},
                "created_at": datetime.now().isoformat(),
            },
            "contents": contents,
            "embedding": embedding,
        }
    )

records_df = df.apply(process_row, axis=1)
vec.create_tables()
vec.create_index()  # DiskAnnIndex
vec.upsert(records_df)

# with ThreadPoolExecutor(max_workers=4) as executor:
#     embeddings = list(executor.map(process_row, df))

