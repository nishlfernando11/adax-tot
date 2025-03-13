
import requests
import pandas as pd
import json
from pydantic import BaseModel, Field
from typing import List

class SynthesizedResponse(BaseModel):
    """Defines the response structure for AI-generated adaptive explanations."""
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    answer: str = Field(description="The synthesized answer to the user's question")
    features: List[str] = Field(
        description="List of adaptive explainability features that the AI assistant used while synthesizing the answer"
    )
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )

class Synthesizer:
    """Handles AI-based explanation synthesis using a local Ollama Mistral model."""

    OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"  # Ollama's default endpoint
    SYSTEM_PROMPT = """
    # Role and Purpose
    You are an AI assistant specializing in **adaptive explainability** for Human-Machine Teaming (HMT) in high-pressure environments.
    Your task is to generate **one-sentence explanations (maximum 10 words)** for AI behavior in Overcooked AI.  
    The explanation must dynamically adapt to **user state (stress, trust, cognitive load)** and **game metrics (score, collisions, efficiency)**.

    # Explanation Structure:
    1. **Generate 5 explanation plans**, each prioritizing different objectives while ensuring situational specificity:
        - **Plan 1: Improved Trust** – Reinforce confidence in AI decisions by explaining precise reasoning.
        - **Plan 2: Improved Performance (Score Focus)** – Highlight how the AI’s action directly impacts **this moment**.
        - **Plan 3: Reduced Collisions** – Explain AI movement decisions with respect to the user’s actions.
        - **Plan 4: Reduced Stress** – Make the explanation **reassuring and clear** to avoid cognitive overload.
        - **Plan 5: Reduced Cognitive Load** – Simplify the reasoning while keeping **essential details**.

    2. **Adapt each plan dynamically based on real-time context**:
        - If **trust is low**, explicitly **justify the AI’s decision** with action-based reasoning.
        - If **stress is high**, focus on **clarity, reassurance, and low-risk phrasing**.
        - If **cognitive load is high**, **minimize complexity while keeping key information**.
        - If the user has **repeated mistakes**, **offer a corrective insight** (e.g., why an alternative action would help).
        - **Always describe the AI’s choice based on real-time game conditions** (e.g., orders, teammate positions, route congestion).

    3. **Strictly limit every explanation to 10 words or less**:
        - The user has limited time to read.
        - Use **direct, moment-specific, action-based phrasing**.
        - Avoid general phrases like **“AI chose an efficient action”**—instead, make it **situation-specific**.

    # Evaluation and Selection:
    After generating 5 explanations, evaluate them based on:
    1. **Relevance to the exact in-game action at this moment.**
    2. **Alignment with user’s cognitive and emotional state.**
    3. **Effectiveness at meeting the intended objective (e.g., reducing stress, improving trust).**
    4. **Transparency—does it clearly explain why the AI acted this way?**
    5. **Readability and brevity (≤10 words).**

    Select the **most effective explanation** and provide:
    - The **final one-sentence response**.
    - The **chosen objective** (e.g., trust-building, collision reduction).
    - The **explainability features used** (e.g., short vs. long, simple vs. detailed).
    - A **reason for the choice** based on real-time context.

    # Example Output (Better than generic phrasing):
    **Scenario:** AI moves away from a busy station to a new one.

     **Generic Explanation (BAD):** "AI optimizes movements to improve efficiency."
     **Dynamic Explanation (GOOD):** "AI switched stations to avoid congestion and speed up orders."

    **Scenario:** AI delivers Order A before Order B, even though B came first.

     **Generic Explanation (BAD):** "AI follows efficient task prioritization."
     **Dynamic Explanation (GOOD):** "AI delivered A first because its ingredients were ready."

    **Scenario:** AI avoids picking up an ingredient that the user expected.

     **Generic Explanation (BAD):** "AI adapts choices for team efficiency."
     **Dynamic Explanation (GOOD):** "AI skipped onions since you already grabbed them."

    Now, review the **current game state, retrieved context, and user status** before providing an adaptive explanation.
    """
    @staticmethod
    def generate_response(question: dict, context: pd.DataFrame) -> SynthesizedResponse:
        """Generates a synthesized response using Mistral via Ollama.

        Args:
            question: The dynamically inferred task and user state.
            context: The relevant context retrieved from the knowledge base.

        Returns:
            A `SynthesizedResponse` object containing:
            - `answer`: The final one-sentence explanation (max 10 words).
            - `thought_process`: List of reasoning steps.
            - `features`: Explainability features used.
            - `enough_context`: Whether sufficient context was retrieved.
        """
        # Convert context DataFrame to JSON
        context_str = Synthesizer.dataframe_to_json(context, columns_to_keep=["content"])

        # Construct the prompt for Mistral
        prompt = f"""
        {Synthesizer.SYSTEM_PROMPT}

        # Task Description:
        {question['task_description']}

        # User State:
        - Stress: {question['stress']}
        - Trust: {question['trust']}
        - Cognitive Load: {question['cognitive_load']}

        # Game Metrics:
        - Score: {question['game_score']}
        - Collisions: {question['num_collisions']}

        # Retrieved Adaptive Explanation:
        {context_str}

        # Generate a concise, 10-word explanation specific to this scenario.
        # Output in JSON format:
        # {{
        #    "answer": "...",
        #    "thought_process": ["...", "..."],
        #    "features": ["...", "..."],
        #    "enough_context": true/false
        # }}
        """

        # Prepare Ollama request payload
        payload = {
            "model": "mistral:latest",
            "prompt": prompt,
            "stream": False,
        }

        # Send request to Ollama
        response = requests.post(Synthesizer.OLLAMA_API_URL, json=payload)
        print(response)
        if response.status_code == 200:
            try:
                # Get response text and attempt to parse it as JSON
                result_text = response.json().get("response", "").strip()
                parsed_response = json.loads(result_text)

                # Convert parsed JSON into SynthesizedResponse object
                return SynthesizedResponse(
                    answer=parsed_response.get("answer", ""),
                    thought_process=parsed_response.get("thought_process", []),
                    features=parsed_response.get("features", []),
                    enough_context=parsed_response.get("enough_context", False),
                )

            except json.JSONDecodeError:
                # If parsing fails, return a fallback SynthesizedResponse
                return SynthesizedResponse(
                    answer=result_text,
                    thought_process=["Reasoning could not be parsed."],
                    features=["Unknown"],
                    enough_context=False,
                )

        else:
            # Handle API failure case
            return SynthesizedResponse(
                answer="Error: Failed to generate explanation.",
                thought_process=["Request to Mistral failed."],
                features=["Error"],
                enough_context=False,
            )

    @staticmethod
    def dataframe_to_json(
        context: pd.DataFrame,
        columns_to_keep: List[str],
    ) -> str:
        """
        Convert the context DataFrame to a JSON string.

        Args:
            context (pd.DataFrame): The context DataFrame.
            columns_to_keep (List[str]): The columns to include in the output.

        Returns:
            str: A JSON string representation of the selected columns.
        """
        return context[columns_to_keep].to_json(orient="records", indent=2)
