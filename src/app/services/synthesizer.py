from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from services.llm_factory import LLMFactory
import json

class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    answer: str = Field(description="The synthesized answer to the user's question")
    features: List[str] = Field(description="List of adaptive explainability features that the AI assistant used while synthesizing the answer")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )


class Synthesizer:
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

    ❌ **Generic Explanation (BAD):** "AI optimizes movements to improve efficiency."
    ✅ **Dynamic Explanation (GOOD):** "AI switched stations to avoid congestion and speed up orders."

    **Scenario:** AI delivers Order A before Order B, even though B came first.

    ❌ **Generic Explanation (BAD):** "AI follows efficient task prioritization."
    ✅ **Dynamic Explanation (GOOD):** "AI delivered A first because its ingredients were ready."

    **Scenario:** AI avoids picking up an ingredient that the user expected.

    ❌ **Generic Explanation (BAD):** "AI adapts choices for team efficiency."
    ✅ **Dynamic Explanation (GOOD):** "AI skipped onions since you already grabbed them."

    Now, review the **current game state, retrieved context, and user status** before providing an adaptive explanation.
    """

    # SYSTEM_PROMPT = """
    # # Role and Purpose
    # You are an AI assistant specializing in **adaptive explainability** for Human-Machine Teaming (HMT) in high-pressure environments.
    # Your task is to generate **one-sentence explanations (maximum 10 words)** for AI behavior in Overcooked AI.  
    # The explanation must dynamically adapt to **user state (stress, trust, cognitive load)** and **game metrics (score, collisions, efficiency)**.

    # # Explanation Structure:
    # 1. Generate **5 explanation plans**, each prioritizing different objectives:
    # - **Plan 1: Improved Trust** – Build user confidence in AI decisions.
    # - **Plan 2: Improved Performance (Score Focus)** – Help the user optimize gameplay.
    # - **Plan 3: Reduced Collisions** – Minimize AI-player movement conflicts.
    # - **Plan 4: Reduced Stress** – Reassure the user and lower cognitive burden.
    # - **Plan 5: Reduced Cognitive Load** – Deliver the simplest, easiest-to-process explanation.

    # 2. **Sub-branch each plan** based on context:
    # - If **trust is low**, use **justifications** for AI behavior.
    # - If **stress is high**, use **reassuring, risk-reducing phrasing**.
    # - If **cognitive load is high**, **use the simplest possible sentence**.

    # 3. **Limit every explanation to 10 words or less**.
    # - The user has limited time to read.
    # - Use direct, action-based phrasing.

    # # Evaluation and Selection:
    # After generating the 5 explanations, evaluate them based on:
    # 1. **Relevance to the current task and retrieved past interactions.**
    # 2. **Alignment with user’s cognitive and emotional state.**
    # 3. **Effectiveness at meeting the intended objective (e.g., reducing stress).**
    # 4. **Transparency and justification for AI behavior.**
    # 5. **Readability and brevity (≤10 words).**

    # Select the **most effective explanation** and provide:
    # - The final **one-sentence response**.
    # - The **chosen objective** (e.g., trust-building, collision reduction).
    # - The **explainability features used** (e.g., short vs. long, simple vs. detailed).
    # - A **reason for the choice** based on real-time context.

    # # Example Output:
    # **Selected Explanation:** "AI avoids collisions by taking the safer route."
    # **Objective:** Reduced Collisions  
    # **Explainability Features:** Short, Simple, Risk-Aware  
    # **Reason:** Player had 3 past collisions, stress level was high.

    # Now, review the **current game state, retrieved context, and user status** before providing an adaptive explanation.
    # """


    @staticmethod
    def generate_response(question: object, context: pd.DataFrame) -> SynthesizedResponse:
        """Generates a synthesized response based on the question and context.

        Args:
            question: The user's question.
            context: The relevant context retrieved from the knowledge base.

        Returns:
            A SynthesizedResponse containing thought process and answer.
        """
        context_str = Synthesizer.dataframe_to_json(
            context, columns_to_keep=["content"] #"category"
        )

        # messages = [
        #     {"role": "system", "content": Synthesizer.SYSTEM_PROMPT},
        #     {"role": "user", "content": f"# User question:\n{question}"},
        #     {
        #         "role": "assistant",
        #         "content": f"# Retrieved information:\n{context_str}",
        #     },
        # ]
        
        messages = [
        {"role": "system", "content": Synthesizer.SYSTEM_PROMPT},
        {
            "role": "user",
            "content" : (
                f"# Task Description:\n{question['task_description']}\n"
                f"# User State:\n- Stress: {question['stress']}\n- Trust: {question['trust']}\n"
                f"- Cognitive Load: {question['cognitive_load']}\n"
                f"# Game Metrics:\n- Score: {question['game_score']}\n- Collisions: {question['num_collisions']}\n"
            ),
        },
        {
            "role": "assistant",
            "content": (
                f"# Retrieved Adaptive Explanation:\n{context_str}"
                # f"- Simplified: {explanation_simplified}\n"
                # f"- Balanced: {explanation_balanced}\n"
                # f"- Step-by-Step: {explanation_step_by_step}"
            ),
        },
        ]

        llm = LLMFactory("openai")
        return llm.create_completion(
            response_model=SynthesizedResponse,
            messages=messages,
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
