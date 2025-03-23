import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMStatePrompter:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32
        ).to(self.device)

    def format_prompt(self, state_list, instruction="Summarize the current Overcooked AI collaboration state and trends in 2 sentences."):
        """
        Accepts a list of 1+ state dictionaries and builds a structured prompt.
        """
        if not isinstance(state_list, list):
            state_list = [state_list]

        state_descriptions = []
        for i, state in enumerate(state_list):
            label = "Current state" if i == len(state_list) - 1 else f"Previous state {i + 1}"
            lines = [f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in state.items()]
            state_block = f"{label}:" + "\n".join(lines)
            state_descriptions.append(state_block)

        all_states = "\n\n".join(state_descriptions)

        prompt = f"""
You are an intelligent assistant analyzing gameplay in a cooperative cooking game (Overcooked).
Your goal is to analyze player coordination, task progress, and interaction dynamics.
Use provided game states (1 or more) to describe:
- Current player behavior and task situation
- Notable trends in performance (e.g. score growth, sudden collision spike)
- Brief interpretation of the AI and user collaboration

{all_states}

Instruction: {instruction}

Provide your analysis summary below:
"""
        return prompt

    def generate_summary(self, state_dict_or_list, instruction="Summarize the current Overcooked AI collaboration state and trends in 2 sentences."):
        prompt = self.format_prompt(state_dict_or_list, instruction)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.6,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result.replace(prompt, "").strip()


# Example usage
if __name__ == "__main__":
    prompter = LLMStatePrompter()
    current = {
        "ai_position": [3, 1],
        "ai_orientation": [1, 0],
        "ai_held_object": None,
        "user_position": [1, 2],
        "user_orientation": [0, -1],
        "user_held_object": None,
        "orders": ["onion", "onion", "onion"],
        "score": 0,
        "collisions": 2,
        "time_left": 23.1,
        "layout_name": "cramped_room",
        "timestep": 38,
        "joint_action": "[[0, 0], [0, 1]]"
    }
    prev = {
        "ai_position": [3, 2],
        "score": 0,
        "collisions": 1,
        "time_left": 24.5,
        "timestep": 37,
        "user_position": [1, 2]
    }
    summary = prompter.generate_summary([prev, current])
    print("\nGenerated Summary (with trends):\n", summary)

    summary = prompter.generate_summary(current)
    print("\nGenerated Summary (single state):\n", summary)
