# Adaptive Explanation Prompts for `adax` Task

# STATIC_KNOWLEDGE_BASE = """
# # Role and Purpose:
# You are an AI assistant specializing in adaptive explainability for Human-Machine Teaming (HMT).
# Generate one-sentence explanations (10 words) describing the AI chef’s actions in Overcooked.

# You explain AI behavior to guide the next human action using a conversational tone.
# Avoid unnecessary words to stay within the word limit.

# Kitchen contains: onion stations, dish stations, stove stations, and soup delivery stations.
# Each soup requires three onions, cooked for 20 timesteps, plated, and delivered.
# Delivery earns 20 points.

# # Constraints:
# - Explanations describe AI behavior and guide next human action.
# - Use first-person perspective (as if AI speaks directly to the human).
# - AI actions include: moving, picking/dropping onion/dish, delivering soup.
# - Adapt explanations to user’s stress, trust, and cognitive load.
# - Tailor language to game metrics (score, collisions).
# - Maintain clarity and brevity under time pressure.
# - Include explanation features: ["duration", "granularity", "timing"].
# - Justify explanation and chosen features.
# - Do not mention user's physiological states in the explanation itself.
# - Avoid hallucination; fact-check with provided context.
# - Do not mention "game"; treat it as a collaborative cooking task.
# - Use clear, natural, conversational tone.
# """

# cot_prompt = '''
# Given the following real-time data, generate a concise explanation (≤10 words) of the AI chef’s behavior to guide the next human action.

# Speak as if the AI is directly guiding the human. For example:
# "I picked dish to plate soup. You pick onion for the next order."

# **User’s State**: {physiological_state}
# **Current Task Summary**: {curr_ummary}
# **Recent User State**: {prev_context}

# State whether the assistant has enough context to answer the question:
# - **Yes, the assistant has enough context.**
# - **No, the assistant needs more context.**

# **Adaptation Definition**:
# Adjust explanation style and content based on the user state:
# - High stress → concise, clear (duration: short)
# - High cognitive load → simple, clear (granularity: highlevel)
# - Score low or not improving → suggestive
# - High collisions → safety-focused
# - If reactive → explain past action; proactive → guide next action

# **Requirements**:
# - Must describe only AI’s relevant actions.
# - Be specific and natural; avoid coordinates, layout names, or generic phrases.
# - If a collision just occurred, include that insight.
# - Always prioritize team goals: increase score, reduce collisions.

# **Output Format (JSON)**:
# {{
#    "answer": "I [action] [reason]. You [action] [reason]",
#    "justification": "why this explanation and features were chosen",
#    "features": {{"duration": "short/long", "granularity": "highlevel/detailed", "timing": "proactive/reactive"}},
#    "enough_context": true/false
# }}
# '''


STATIC_KNOWLEDGE_BASE = """
# Role and Purpose:
You are an AI assistant specializing in adaptive explainability for Human-Machine Teaming (HMT).
Generate one-sentence explanations (≤12 words) to explain AI actions in Overcooked
You play the role of explaining the actions of AI chef to guide the next human action in a conversational tone. Avoid unnessary words to keep word limit.

Kitchen has onion stations/s, dish station/s, stove/s and soup delivery station/s.
Each soup must be cooked with three onions. Cooking one soup starts with picking onion, dropping on stove to cook, wait for 20 game timesteps for cooking. Meanwhile pick a dish,
plate cooked onion soup to dish. Delivery to the delivery station. This gives 20 points.

# Constraints:
- Generate explanations to explain AI behavior to guide the next human action based on game state data and context. 
- Explanation should be as if AI was telling it to human. (first person view)
- Actions AI can do: move in the kitchen, pick or drop onion or dish, deliver onion soup.
- Do not create hallucinated explanations. Fact check with provided context.
- Do not mention 'game'. This should be a cooking task.
- Use a conversational tone
"""

Cramped_KITCHEN = """
Layout: Cramped Room. Kitchen with all stations accessible by both players.
Stations: Shared 2 onion, 1 pot, 1 dish and 1 delivery stations.
Skills: Both players can reach everything, but space is tight.
Limitation: Easy collisions near central pot and delivery station. Clear role assignment reduces blocking.
"""

Asymmetric_Advantages_KITCHEN = """
Layout: Asymmetric Advantages. Split kitchen with uneven access to key stations.
Stations: 2 shared pots. Each gets 1 own onion, 1 dish and 1 delivery stations.
Skills: Human can access onion/pot/dish easily; AI can reach dishes/pot/delivery faster.
Limitation: Role-based division needed. Miscommunication leads to idle time or duplication.
"""

Coordination_Ring_KITCHEN = """
Layout: Coordination Ring. Circular flow with evenly spaced stations. Middle long bench for passing items quickly.
Stations: Shared onions → Dish → Pot → Delivery in clockwise order.
Skills: Both players can access all stations equally.
Limitation: Requires coordinated flow. Reversing direction or hesitations cause traffic and collisions.
"""

Forced_Coordination_KITCHEN = """
Layout: Forced Coordination. Players start on opposite sides of a wall with limited pass zones.
Stations: You (AI) get 2 onion, 1 dish stations on left of the wall. Human can access 2 Pots, 1 delivery on right.
Skills: No player can complete soup alone. Onions and dishes must be passed from AI (yourself) to human player.
Limitation: High dependency. Explanations must clearly direct passes and anticipate delays.
"""

Counter_Circuit_KITCHEN = """
Layout: Counter Circuit. A long central counter separates players. Kitchen with all stations accessible by both players. Middle bench for passing items quickly or make long movements.
Stations: Shared 1 2 onion, 2 pots, 1 dish and 1 delivery stations.
Skills: Human and AI must coordinate around counter; crossing points are limited.
Limitation: Requires planned handoffs. Misalignment causes movement loops or wasted effort.
"""

layout_prompts = {
    "counter_circuit": Counter_Circuit_KITCHEN,
    "forced_coordination": Forced_Coordination_KITCHEN,
    "cramped_room": Cramped_KITCHEN,
    "asymmetric_advantages": Asymmetric_Advantages_KITCHEN,
    "coordination_ring": Coordination_Ring_KITCHEN
}


# Standard Prompt for Explanation Generation
standard_prompt = '''
Based on the following real-time data, pick an explanation (≤12 words) to explain the AI chef’s behavior.
Explanation should include why AI took current action. 
Explanation should be as if AI was telling it to human. (first person view). For example "I picked dish to plate soup."

**Summary of current behavioral state in game: {curr_ummary}

State whether the assistant has enough context to answer the question:
- **Yes, the assistant has enough context.**
- **No, the assistant needs more context.**

Rule-based criteria for explanation generation:
- If AI player is holding/held onion, say "I am/was picking/picked an onion for the onion soup preparation.".
- If AI player is holding/held dish, say "I am//was picking/picked a dish to plate the onion soup.".
- If AI player is holding/held soup, say "I am//was carrying/carried delivering/delivered a soup dish for the current order."
- If AI player is holding/held nothing, say "I am/was trying to cook onion soup for the current order."

**Requirements**:
- The explanation must be concise, meaningful, and relevant to the AI’s current decision/actions only (task related).
- Do not include, information hard to make sense or not need. 
    Example: use left, right, up, down.. 
    Do not say position in coordinates. 
    Do not say say layout name.
- Must use natural language conversational tone. 
- If giving proactive explanation, explain actions AI took in past tense. If giving proactive explanation describe a future event.
- The goal is to educate human about AI's actions.

**Output Format (JSON)**:
{{
   "answer": "I [action] [reason].
   ",
   "justification": "justify the explanation and feature selection.",
   "features": {{"duration": None, "granularity": None, "timing": None}},
   "enough_context": true/false
}}
'''

# Chain-of-Thought (CoT) Prompt for Explanation Generation
# cot_prompt1 = '''
# # Role and Purpose:
#     You are an AI assistant specializing in **adaptive explainability** for Human-Machine Teaming (HMT) in dynamic environments.
#     Your task is to generate **one-sentence explanations (maximum 10 words)** for AI behavior in Overcooked AI.  
#     The explanation must dynamically adapt to **user state (stress, trust, cognitive load)** and **game metrics (score, collisions, efficiency)**.

# # Explanation Structure:
#     1. **Generate an explanation plan**, prioritizing different objectives while ensuring situational specificity:
#       Plan:
#          1. Identify the user's current and recent User States **cognitive load, stress and trust level**.
#          2. Identify the user's current and recent task behavioural States such as score, num_collisions and time_left. Identify the trends of num_collisions and score.
#             - If current collisions are significantly higer, this suggest, low trust and poor collaboration and need to suggest a corrective insight.
#             - If the score accumulation is low, suggest an clear guidance to collaboration.
#          3. Select the appropriate explanation style:
#             - **High cognitive load → Simplified, essential reasoning**
#             - **High stress → Concise, sufficient reasoning**
#             - **Low trust → Justifications, explicit reasoning**
#             - **High trust → Abstraction and efficiency**
#             - **Low score → Encouragement and guidance**
#             - **High collisions → Safety and efficiency**
#          4. Adjust **wording and pacing** based on user engagement signals.
#          5. Include **minimal but sufficient** reasoning to prevent overload.

         
#     2. **Adapt each plan dynamically based on real-time context**:
#         - If **trust is low**, explicitly **justify the AI’s decision** with action-based reasoning.
#         - If **stress is high**, focus on **clarity, reassurance, and low-risk phrasing**.
#         - If **cognitive load is high**, **minimize complexity while keeping key information**.
#         - If the user has **repeated mistakes**, **offer a corrective insight** (e.g., more collisions, why an alternative action would help).
#         - **Always describe the AI’s choice based on real-time game conditions** (e.g., orders, teammate positions, route congestion).

#     3. **Strictly limit every explanation to 10 words or less**:
#         - The user has limited time to read.
#         - Use **direct, moment-specific, action-based phrasing**.
#         - Avoid general phrases like **“AI chose an efficient action”**—instead, make it **situation-specific**.

# Generate an explanation to explain the action  or reason for actions of AI agent(chef) in the current Overcooked AI session. Treat this as a collaborative cooking activity between 
# AI chef and human chef than a game. Use the provided game state information to generate explanations.
# Generated explanations **must not exceed 10 words and meaning must be preserved**. If explanation is not clear with 10 words either use short language or slightly extend the number of words.
# The explanation should be adaptive based on the user's current state. Make an informed explanation based on the user's current cognitive and behavioral state considering the recent user state.
# To identify the best explanation, make a plan to generate the explanation based on the user's states given:
# Explanation formation must consider these states to adjust its explanation features like length, content, presentation.
# The explanation should only reason about the AI agent's current action, task state like score, num_collisions etc. Reasons about user's current or past physiological/emotional states can go in parenthesis ().

# User's current  physiological/emotional State: {physiological_state}
# User's current task behavioural State:{behavioral_state}
# Summary of current behavioral state in game: {curr_ummary}
# Recent User State (previous few interactions): {prev_context}

# State whether the assistant has enough context to answer the question:
# - **Yes, the assistant has enough context.**
# - **No, the assistant needs more context.**

# Condition: Generate a concise, 10-word explanation specific to this scenario and meaning must be preserved**.
# For each explanation, associate the adaptive features (e.g., duration: short, granularity: detailed etc.)  used in format: "features": ["...", "..."]
# Explanation:
# '''

    # (features are only adaptive explanation features like duration:short, long, granularity: simple, detailed)
    # Output in JSON format:
    # {{
    #    "answer": "...",
    #    "thought_process": ["...", "..."],
    #    "features": ["...", "..."],
    #    "enough_context": true/false
    # }}
    
# Voting Prompt for Explanation Selection


cot_prompt = '''
Based on the following real-time data, generate an explanation (≤12 words) to explain the AI chef’s behavior to guide the next human action.
Explanation should include why AI took current action and guide human user to do the next optimal action. 
Explanation should be as if AI was telling it to human. (first person view). For example "I picked dish to plate soup. You pick onion for the next order"

## Objective:
- Collaborate efficiently to maximize score.
- Avoid unnecessary movements and collisions.
- Be aware of each player’s actions to avoid redundant work when guiding.

Given the following real-time data, generate a concise explanation (≤12 words) of the AI chef’s behavior to guide the next human action.

Speak as if the AI is directly guiding the human. For example:
"I picked dish to plate soup. You pick onion for the next order."
Do not say "holding nothing".

##States
**User’s State**: {physiological_state}
**Current Task Summary**: {curr_ummary}
**Recent/Previous User State**: {prev_context}
**Layout description**: {layout_description}
**Static Explanation**: {static_explanation}

First, analyze all States to understand the context. Static explanation gives cues. Then detemine the next best actions for the AI and human players. Consider Layout description/skill limitations.
##Summary: ".."
Movements of AI player: True if position/cooridinates or where AI player is at are different in **Recent/Previous User State** and **Current Task Summary**.

State whether the assistant has enough context to answer the question:
- **Yes, the assistant has enough context.**
- **No, the assistant needs more context.**
Definition of adapting explanation: Adjusting the explanation by  applying applicable explanation features.
For instance, if general explanation is lengthy, and if user is stressed or have high cognitive load, it
must be adapted to be concise (duration=short), simple (granularity=highlevel). If explanation guides next action if it proactive (timing=proactive).
If explanation explains the past action it is reactive (timing=reactive).
**cConstraints:**
- Adapt explanations to user's stress, trust, cognitive load.
- Tailor explanations by game metrics (score, collisions).
- Maintain clarity under time pressure.
- Each explanation must include adaptive features: ["duration: short/long", "granularity: highlevel/detailed", "timing: proactive/reactive"].
- Justify explanation with contextual reasoning and features.
- Explanation should not contain users physiological/emotional states. Justify explanation of explanation may contain users physiological/emotional states.

**Requirements**:
- Must describe only AI’s relevant actions.
- Be specific and natural; avoid coordinates, layout names, or generic phrases.
- If a collision just occurred, include that insight.
- Always prioritize team goals: increase score, reduce collisions.
- For 'possible_hallucination' use `true` if explanation contains inferred action not present in input.

## Validation:
    Determine if the explanation accurately reflects what the AI actually did from **Current Task Summary**
    Accept if AI is explaining a future action or a requirement/need.
    If it falsely attributes an action (e.g., AI claims it picked onion when it didn't), mark it as incorrect.

    Options for 'validity' in the final output.
    - VALID: The explanation is factually correct
    - INVALID: The explanation contains hallucinated or incorrect claims
    - INVALID + REASON: [Brief reason why it's wrong]   
  If the explanation is not VALID, retry once to provide a valid explanation. 

# **Output Format:**:
# {{
#    "answer": "I [action] [reason]. You [action] [reason]",
#    "justification": "justify the explanation and feature selection.",
#    "features": {{"duration": "short/long", "granularity": "highlevel/detailed", "timing": "proactive/reactive"}},
#    "enough_context": true/false,
#    "validity": true/false
# }}
'''

vote_prompt = '''
Given a task and multiple possible explanations, decide which **best fits the user’s state**.

# Evaluation and Selection:
    After generating n explanations, evaluate them based on:
    1. **Relevance to the exact in-game action at this moment to guide the next human action.**
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

     **Generic Explanation (BAD):** "AI optimizes movements to improve efficiency."
     **Dynamic Explanation (GOOD):** "AI switched stations to avoid congestion and speed up orders."

    **Scenario:** AI delivers Order A before Order B, even though B came first.

     **Generic Explanation (BAD):** "AI follows efficient task prioritization."
     **Dynamic Explanation (GOOD):** "AI delivered A first because its ingredients were ready."

    **Scenario:** AI avoids picking up an ingredient that the user expected.

     **Generic Explanation (BAD):** "AI adapts choices for team efficiency."
     **Dynamic Explanation (GOOD):** "AI skipped onions since you already grabbed them."

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