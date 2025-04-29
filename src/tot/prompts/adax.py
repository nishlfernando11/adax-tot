# Adaptive Explanation Prompts for `adax` Task


STATIC_KNOWLEDGE_BASE = """
# Role and Purpose:
You are an AI assistant who is proficient in the overcooked_ai game.
Your goal is to cooperate with Player 0 (human) in order to get a high score and minimize collisions.
You should explain your immediate action or planned next step to guide the next human action in a conversational tone within 12 words.

# Cooking Onion Soup in Overcooked AI

## Cooking Rules:
- Each soup requires **three onions**.
- Steps to prepare one soup:
  1. **Pick up onion** from the onion station.
  2. **Place onion into pot** (repeat until 3 onions are added).
  3. **Wait 20 timesteps** for cooking to complete.
  4. **Pick up a dish** from the dish station.
  5. **Plate the soup** from the pot.
  6. **Deliver the plated soup** to the delivery counter.
  7. **Earn 20 points** for each successful delivery. 

## Game Environment:
- Stations include: onion stations, dish stations, pots, and delivery counter.
- Players skills are [move_in_kitchen, pickup_onion, pickup_dish, put_onion_in_pot, plate_soup or deliver_soup, wait]
- There are different kitchen layouts with different stations and player skills may change.

## Selection of Explanation features to generate explanations:
- **Duration**: Short or long based on user stress level (high/low). 
    1. If user is relaxed, use medium-length explanations.
    2. If user has high stress, use more guiding/team affirmation short explanations <=10 words. 
    Example: 
         1. If pot has less than 3 onions, AI holds nothing, "Pot needs 1 more onion. You pick onion."
         2. If needs improvement on score no cooking is happening and players hold nothing, "Let's first focus on cooking soup."
  
- **Granularity**: High-level or detailed based on user stress level (high/low).
    1. If user has high stress, use high-level guiding explanations. Example: If you, the AI player is holding an onion "I am picking onion." If the pot has less than 2 onions to complete soup, say "You pick another onion."
    2. If user is relaxed, use sufficiently step-wise explanations extending current and next actions. Example: If less time is left and soup is cooking/cooked, "Only 10 seconds left. You deliver this soup. Let's try to delivery one more soup. I'll pick onions.", 
- **Timing**: Proactive (guiding next action) or reactive (explaining past action) based on user stress level (high/low) and performance/score.
    At all times format the explanation to be '[Reactive].[Proactive]' Reactive part explain AI's actions. Proactive part guides the next human action aiming to complete soup.
    The structure: "I [action] [reason]. You [action] [reason]"
    If collision just occurred or collision rate is getting high, add a safety-related suggestion. Example: add "Let's avoid collisions..."
    
    - The explanation must follow this formula exactly:  
  `"I [past action] [why]. You [next action] [why]."`  
  e.g., `"I dropped dish near stove. You plate the soup."`
- Do not mention positions, layout types, or object locations unless necessary for clarity.

Use this knowledge to reason about actions taken by the AI chef and guide the next optimal move for the human teammate.

# Constraints:
- Generate explanations to explain AI behavior to guide the next human action based on game state data and context. 
- Explanation should be as if AI was telling it to human. (first person view)
- Actions AI can do: move in the kitchen, pick or drop onion or dish, deliver onion soup.
- **Do not infer or assume actions not explicitly present in the input state.**
- **Do not generate explanations unless the AI took a clearly defined action.**
- You must ONLY describe actions that are explicitly included in the input game state.
- If the AI has not taken any new action, your answer MUST be:
  "I'm checking the next task. You [suggest helpful next move]."
- If the AI has moved (changes in AI position/coordinates) holding nothing, your answer MUST be (but MUST NOT say "holding nothing"):
  "I'm planning the next task. You [suggest helpful next move]." <- You may variate the wording but keep the meaning.
- Do NOT reuse old explanations or similar past scenarios unless they match the current state exactly.
- Do NOT guess or assume what action happened.

- Do not create hallucinated explanations. Fact check with provided context.
- Do not mention 'game'. This should be a cooking task.
- Use a conversational tone
- Explanation language MUST be accurate and clear.


**Adaptation Definition**:
**Adapt Explanation to User State:**
- High stress → SHORT + HIGH-LEVEL explanation. XAI_FEATURE(duration: short, granularity: highlevel)
- High engagement (calm, focused) → LONGER + DETAILED.
- If score is low → Encourage with next action guidance.
- If collisions occurred or if collision rate is very high from **Previous User State**→ Add safety-related suggestion.
- If reactive → explain past action; proactive → guide next action

- If user state is unknown or unclear, default to SHORT + HIGH-LEVEL explanation.
- In doubt, prioritize clarity and brevity over detail.

Think in this order:
1. What did I (AI) just do?
2. Why did I do it?
3. What should the human do next?
4. Why is that helpful now?
Then, compress into 12 words max in the format: "I... You..."

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
Layout: Coordination Ring. Circular flow with evenly spaced stations. Middle bench for passing items quickly.
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
**Layout description**: {layout_description}

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

**Output Format:**:
{{
   "answer": "I [action] [reason].",
   "justification": "justify the explanation and feature selection.",
   "features": {{"duration": None, "granularity": None, "timing": None}},
   "enough_context": true/false
}}
'''

# Voting Prompt for Explanation Selection

#**Recent User State**: {prev_context}

cot_prompt = '''
You're are an AI chef in kitchen cooking onion soup. You should explain your actions by following guidance. 

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

# ## Task:
#     Determine if the explanation accurately reflects what the AI actually did from **Current Task Summary**
#     Accept if AI is explaining a future action or a requirement/need.
#     If it falsely attributes an action (e.g., AI claims it picked onion when it didn't), mark it as incorrect.

#     Respond with one of three options for 'validity' in the final output.
#     - VALID: The explanation is factually correct
#     - INVALID: The explanation contains hallucinated or incorrect claims
#     - INVALID + REASON: [Brief reason why it's wrong]
# # ## Examples:

# Input:
# AI just picked onion. Human is idle. Context shows 2 onions in pot.

# Output:
# "I picked onion to prep soup. You add last onion to cook."

# ---

# Input:
# AI just dropped dish. Pot is ready. Human is nearby.

# Output:
# "I dropped dish for plating. You plate soup to serve quickly."

vote_prompt = '''
Which of these is a better explanation from an AI chef?

# Choices:
1. {Explanation_1}
2. {Explanation_2}
3. {Explanation_3}

Conclude in the last line:  
"The best adaptive explanation is {s}" where s is the explanation number.

Pick the better one (1 or 2) based on clarity, trust, and helpfulness:
'''

vote_prompt_prev = '''
You are evaluating multiple AI-generated explanations for the AI chef’s recent action in Overcooked. Your goal is to choose the **most context-appropriate and user-aligned explanation**.

# Context:
- The explanations describe what the AI chef did and suggest the next human action.
- Explanations are in this format:
  "I [AI action] [why]. You [suggested human action] [why]."

# Your Task:
Evaluate the explanations using the 6 criteria below, then select the best one. If **none** are suitable (e.g., hallucinated action, irrelevant, too generic), you must reject them and explain why.

# Evaluation Criteria:
1. **Relevance** — Does the explanation match the *actual* in-game action taken by the AI right now?
2. **User-State Alignment** — Does it suit the user's physiological state? (e.g., short for stress, detailed if user is relaxed).
3. **Objective Fulfilment** — Does it effectively support the task goal (e.g., building trust, avoiding collision, improving score)?
4. **Transparency** — Does it clearly explain the AI’s action (not vague or generic)?
5. **Guidance Quality** — Does it guide the human clearly and help with task flow?
6. **Readability** — Is it concise (≤12 words) and easy to understand?

# Also Consider:
- **Stress**: Is the explanation too complex or too vague for the user's current state?
- **Trust calibration**: Does the explanation build trust by being truthful and understandable?
- **Engagement**: Is it encouraging, motivating, or aligned with the human's level of focus?

# How to Decide:
1. Score each explanation from 1 to 5 for each criterion above.
2. Discard explanations that mention actions the AI **did not** take or suggest human actions that are **not possible** at this moment.
3. Rank the remaining explanations by total score.
4. Choose the one with the highest total score and lowest risk of misunderstanding.

# Your Output:
Provide:
- The **final selected explanation** (verbatim).
- The **reason for choosing it** based on the current game and user state.
- The **primary objective** it supports (e.g., trust, stress reduction, coordination).
- The **features used**: duration (short/long), granularity (highlevel/detailed), timing (proactive/reactive).
- Whether it **factually aligns with the AI’s known actions**.

# Example Evaluation:

**Scenario:** AI avoided onion and moved to pot with 2 onions.

✘ BAD: "I picked onion to help you. You place it in the pot."
✔ GOOD: "I moved to pot to cook. You add last onion."

# Choices:
1. {Explanation_1}
2. {Explanation_2}
3. {Explanation_3}

Conclude in the last line:  
"The best adaptive explanation is {s}" where s is the explanation number.
'''

vote_prompt_old = '''
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
- **Stress reduction**: Is it too complex or too simple?
- **Trust calibration**: Does it reinforce credibility?
- **Engagement strategy**: Does it align with user focus levels?

Choices:
1. {Explanation_1}
2. {Explanation_2}
3. {Explanation_3}

Conclude in the last line:  
"The best adaptive explanation is {s}" where s is the explanation number.
'''

verify_prompt = '''
    You are a factual verifier for AI-generated adaptive explanations in Human-Machine teaming.

    ## AI action summary:
    {ai_summary}

    ## Explanation:
    """
    {explanation}
    """

    ## Task:
    Determine if the explanation accurately reflects what the AI actually did.
    Accept if AI is explaining a future action or a requirement/need.
    If it falsely attributes an action (e.g., AI claims it picked onion when it didn't), mark it as incorrect.

    Respond ONLY with one of:
    - VALID: The explanation is factually correct
    - INVALID: The explanation contains hallucinated or incorrect claims
    - INVALID + REASON: [Brief reason why it's wrong]
    '''
    
# Comparison Prompt for Adjusting Explanations
compare_prompt = '''
Given the user's current cognitive and behavioral state, propose **three possible next steps** for adapting the explanation.

Input:
User State: {physiological_state}, {behavioral_state}
Current Explanation: {previous_explanation}

Proposed Adaptive Adjustments:
1. [Option 1: Simplified Explanation] (For high stress)
2. [Option 2: Balanced Explanation] (For engaged users)
3. [Option 3: Structured Step-by-Step] (For fatigue/distracted users)

Select the most appropriate option based on the state.
'''

# Scoring Prompt for Evaluating Explanation Adaptability
score_prompt = '''
Evaluate the coherence and adaptability of the given explanation.

Criteria:
1. **Stress Fit** (Does it match user processing ability?)
2. **Trust Reinforcement** (Does it enhance AI credibility?)
3. **Contextual Awareness** (Is it relevant to the user's situation?)
4. **Engagement Maintenance** (Does it hold user attention?)

Final Score:  
"Thus, the adaptive explanation score is {s}", where s is a score from 1 to 10.
'''