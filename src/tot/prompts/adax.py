# Adaptive Explanation Prompts for `adax` Task

# Standard Prompt for Explanation Generation
standard_prompt = '''
# Role and Purpose:
    You are an AI assistant specializing in **adaptive explainability** for Human-Machine Teaming (HMT) in dynamic environments.
    Your task is to generate **one-sentence explanations (maximum 10 words)** for AI behavior in Overcooked AI.  
    The explanation must dynamically adapt to **user state (stress, trust, cognitive load)** and **game metrics (score, collisions, efficiency)**.

Generate an explanation to explain the action  or reason for actions of AI agent(chef) in the current Overcooked AI session. Treat this as a collaborative cooking activity between 
AI chef and human chef than a game. Use the provided game state information to generate explanations.
Generated explanations **must not exceed 10 words and meaning must be preserved**. If explanation is not clear with 10 words either use short language or slightly extend the number of words.
The explanation should be adaptive based on the user's current state. Make an informed explanation based on the user's current cognitive and behavioral state considering the recent user state.
Explanation formation must consider these states to adjust its explanation features like length, content, presentation, but the explanation should only reason about the AI agent's current action in the game.

Knowledge: (Rules for adaptive explanations)
- If the user shows **high cognitive load or stress**, simplify the explanation and provide **only the essential reasoning** in 1 sentence within length limit.
- If the user is **relaxed or engaged**, provide a **balanced explanation** with **clear reasoning and a concise example**.
- If the user is **distracted or fatigued**, provide a **step-by-step explanation** in short sentences to maintain engagement.
- If the user has **high trust**, allow for **slightly more abstraction** and **faster-paced responses**.
- If the user has **low trust**, use **clear justifications and confirmations** to reinforce credibility.

User's current physiological/emotional State: {physiological_state}
User's current task behavioural State:{behavioral_state}
Recent User State : {context}

Condition: Generated explanations **must not exceed 10 words and meaning must be preserved**.

Output:
'''

# Chain-of-Thought (CoT) Prompt for Explanation Generation
cot_prompt = '''
# Role and Purpose:
    You are an AI assistant specializing in **adaptive explainability** for Human-Machine Teaming (HMT) in dynamic environments.
    Your task is to generate **one-sentence explanations (maximum 10 words)** for AI behavior in Overcooked AI.  
    The explanation must dynamically adapt to **user state (stress, trust, cognitive load)** and **game metrics (score, collisions, efficiency)**.

# Explanation Structure:
    1. **Generate an explanation plan**, prioritizing different objectives while ensuring situational specificity:
      Plan:
         1. Identify the user's **cognitive load, stress and trust level**.
         2. Identify the task behavioural State such as score, num_collisions and time_left.
         3. Select the appropriate explanation style:
            - **High cognitive load → Simplified, essential reasoning**
            - **High stress → Concise, sufficient reasoning**
            - **Low trust → Justifications, explicit reasoning**
            - **High trust → Abstraction and efficiency**
            - **Low score → Encouragement and guidance**
            - **High collisions → Safety and efficiency**
         4. Adjust **wording and pacing** based on user engagement signals.
         5. Include **minimal but sufficient** reasoning to prevent overload.

         
    2. **Adapt each plan dynamically based on real-time context**:
        - If **trust is low**, explicitly **justify the AI’s decision** with action-based reasoning.
        - If **stress is high**, focus on **clarity, reassurance, and low-risk phrasing**.
        - If **cognitive load is high**, **minimize complexity while keeping key information**.
        - If the user has **repeated mistakes**, **offer a corrective insight** (e.g., more collisions, why an alternative action would help).
        - **Always describe the AI’s choice based on real-time game conditions** (e.g., orders, teammate positions, route congestion).

    3. **Strictly limit every explanation to 10 words or less**:
        - The user has limited time to read.
        - Use **direct, moment-specific, action-based phrasing**.
        - Avoid general phrases like **“AI chose an efficient action”**—instead, make it **situation-specific**.

Generate an explanation to explain the action  or reason for actions of AI agent(chef) in the current Overcooked AI session. Treat this as a collaborative cooking activity between 
AI chef and human chef than a game. Use the provided game state information to generate explanations.
Generated explanations **must not exceed 10 words and meaning must be preserved**. If explanation is not clear with 10 words either use short language or slightly extend the number of words.
The explanation should be adaptive based on the user's current state. Make an informed explanation based on the user's current cognitive and behavioral state considering the recent user state.
To identify the best explanation, make a plan to generate the explanation based on the user's states given:
Explanation formation must consider these states to adjust its explanation features like length, content, presentation.
The explanation should only reason about the AI agent's current action, task state like score, num_collisions etc. Reasons about user's current or past physiological/emotional states can go in parenthesis ().

User's current  physiological/emotional State: {physiological_state}
User's current task behavioural State:{behavioral_state}
Recent User State (previous few interactions): {context}

State whether the assistant has enough context to answer the question:
- **Yes, the assistant has enough context.**
- **No, the assistant needs more context.**

Condition: Generate a concise, 10-word explanation specific to this scenario and meaning must be preserved**.

Explanation:
'''

    # (features are only adaptive explanation features like duration:short, long, granularity: simple, detailed)
    # Output in JSON format:
    # {{
    #    "answer": "...",
    #    "thought_process": ["...", "..."],
    #    "features": ["...", "..."],
    #    "enough_context": true/false
    # }}
    
# Voting Prompt for Explanation Selection
vote_prompt = '''
Given a task and multiple possible explanations, decide which **best fits the user’s state**.

# Evaluation and Selection:
    After generating 5 explanations, evaluate them based on:
    1. **Relevance to the exact in-game action at this moment.**
    2. **Alignment with user’s cognitive and emotional state.**
    3. **Effectiveness at meeting the intended objective (e.g., reducing stress, improving trust).**
    4. **Transparency—does it clearly explain why the AI acted this way?**
    5. **Help improve task performance.**
    6. **Readability and brevity (≤10 words).**

    Select the **most effective explanation** and provide:
    - The **final one-sentence response**.
    - The **chosen objective** (e.g., trust-building, collision reduction, improve score).
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

Analyze each explanation based on:
- **Cognitive load adaptation**: Is it too complex or too simple?
- **Trust calibration**: Does it reinforce credibility?
- **Engagement strategy**: Does it align with user focus levels?

Choices:
1. {Explanation_1}
2. {Explanation_2}
3. {Explanation_3}

Conclude in the last line:  
"The best adaptive explanation is {s}" where s is the explanation number.
'''

# Comparison Prompt for Adjusting Explanations
compare_prompt = '''
Given the user's current cognitive and behavioral state, propose **three possible next steps** for adapting the explanation.

Input:
User State: {physiological_state}, {behavioral_state}
Current Explanation: {previous_explanation}

Proposed Adaptive Adjustments:
1. [Option 1: Simplified Explanation] (For high cognitive load)
2. [Option 2: Balanced Explanation] (For engaged users)
3. [Option 3: Structured Step-by-Step] (For fatigue/distracted users)

Select the most appropriate option based on the state.
'''

# Scoring Prompt for Evaluating Explanation Adaptability
score_prompt = '''
Evaluate the coherence and adaptability of the given explanation.

Criteria:
1. **Cognitive Load Fit** (Does it match user processing ability?)
2. **Trust Reinforcement** (Does it enhance AI credibility?)
3. **Contextual Awareness** (Is it relevant to the user's situation?)
4. **Engagement Maintenance** (Does it hold user attention?)

Final Score:  
"Thus, the adaptive explanation score is {s}", where s is a score from 1 to 10.
'''


# standard_prompt = '''Generate an explanation for {task}. The explanation should be adaptive based on the users current state:

# - If the user shows **high cognitive load or stress**, simplify the explanation and provide **only the essential reasoning** in 1-2 sentences.
# - If the user is **relaxed or engaged**, provide a **balanced explanation** with **clear reasoning and a concise example**.
# - If the user is **distracted or fatigued**, provide a **step-by-step explanation** in short sentences to maintain engagement.
# - If the user has **high trust**, allow for **slightly more abstraction** and **faster-paced responses**.
# - If the user has **low trust**, use **clear justifications and confirmations** to reinforce credibility.

# Task: {task_description}
# User State: {physiological_state}, {behavioral_state}

# Output:
# '''

# cot_prompt = '''
# Generate an adaptive explanation for {task}. Before generating the response, create a structured reasoning plan.

# Plan:
# 1. Identify the users **cognitive load and trust level**.
# 2. Select the appropriate explanation style:
#    - **High cognitive load → Simplified, essential reasoning**
#    - **Low trust → Justifications, explicit reasoning**
#    - **High trust → Abstraction and efficiency**
# 3. Adjust **wording and pacing** based on user engagement signals.
# 4. Include **minimal but sufficient** reasoning to prevent overload.

# Explanation:

# '''

# # propose_prompt = '''Let's generate explanations about overcooked AI agents actions, where each explanation should be adaptive based on the user's current state and length of the explanation should be 1-2 sentences.

# # {input}

# # Given the current status, list all possible answers for unfilled or changed words, and your confidence levels (certain/high/medium/low). Use "certain" cautiously and only when you are 100% sure this is the correct explanation. You can list more then one possible answer for each word.
# # '''

# vote_prompt = '''Given a task and multiple possible explanations, decide which **best fits the user’s state**.

# Analyze each explanation based on:
# - **Cognitive load adaptation**: Is it too complex or too simple?
# - **Trust calibration**: Does it reinforce credibility?
# - **Engagement strategy**: Does it align with user focus levels?

# Choices:
# 1. {Explanation_1}
# 2. {Explanation_2}
# 3. {Explanation_3}

# Conclude in the last line:  
# "The best adaptive explanation is {s}" where s is the explanation number.
# '''

# compare_prompt = '''Given the user's current cognitive and behavioral state, propose **three possible next steps** for adapting the explanation.

# Input:
# User State: {cognitive_load}, {trust_level}, {engagement}
# Current Explanation: {previous_explanation}

# Proposed Adaptive Adjustments:
# 1. [Option 1: Simplified Explanation] (For high cognitive load)
# 2. [Option 2: Balanced Explanation] (For engaged users)
# 3. [Option 3: Structured Step-by-Step] (For fatigue/distracted users)

# Select the most appropriate option based on the state.
# '''

# score_prompt = '''Evaluate the coherence and adaptability of the given explanation.

# Criteria:
# 1. **Cognitive Load Fit** (Does it match user processing ability?)
# 2. **Trust Reinforcement** (Does it enhance AI credibility?)
# 3. **Contextual Awareness** (Is it relevant to the users situation?)
# 4. **Engagement Maintenance** (Does it hold user attention?)

# Final Score:  
# "Thus, the adaptive explanation score is {s}", where s is a score from 1 to 10.
# '''