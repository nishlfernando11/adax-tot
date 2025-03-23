import os
import re
import json
import subprocess
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.adax import *
from tot.models import gpt
from app import search as searchService
import pandas as pd


class AdaXTask(Task):
    """
    Input (x): Task description + user state (gameplay & physiological data).
    Output (y): Adaptive explanation tailored to stress, trust, and cognitive load.
    Reward (r): Adaptive explanation score based on coherence and trust calibration.
    """

    def __init__(self, file='data.json'):
        """
        file: JSON file containing game metrics, physiological & behavioral data.
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'adax', file) #TODO: load data live
        with open(path, 'r') as f:
            self.data = json.load(f)  # Load task data
        self.steps = 3  # Multi-step generation, evaluation, selection
        # self.stops = [None]  # Stop conditions for explanations
        self.stops = [None] * self.steps  # Ensure stops has `steps` elements


    def __len__(self) -> int:
        return len(self.data)

    # NOTE: Loading live data from a CSV now, TODO: update this method to fetch from streamed data
    def get_input(self, idx: int) -> str:
        """Fetches task description and user state from the dataset."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return json.dumps(self.data[idx], indent=4)  # Convert dict to formatted string

    def run_ollama(self, prompt: str, model="mistral"):
        """Executes Ollama with Mistral to generate explanations."""
        try:
            result = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True)
            return result.stdout.strip()
        except Exception as e:
            print(f"Error running local LLM: {e}")
            return "Error"

    def test_output(self, idx: int, output: str):
        """Evaluates the adaptiveness of an explanation using Mistral via Ollama."""
        entry = self.data[idx]
        
        prompt = score_prompt.format(
            # task_description=entry["task_description"],
            physiological_state=entry["physiological_state"],
            behavioral_state=entry["behavioral_state"]
        ) + output

        score_outputs = [self.run_ollama(prompt) for _ in range(3)]  # Get multiple responses

        scores = []
        for score_output in score_outputs:
            pattern = r".*adaptive explanation score is (\d+).*"
            match = re.match(pattern, score_output, re.DOTALL)
            if match:
                score = int(match.groups()[0])
                scores.append(score)
            else:
                print(f'------------------score no match: {[score_output]}')

        output_limited = ' '.join(output.split()[:10])  # Limit response to 10 words
        info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0, 'final_explanation': output_limited}
        return info

    @staticmethod
    def prepare_context(state: dict) -> pd.DataFrame:
        """Prepares the context features for generating adaptive explanations."""
        # Current user and game state
        current_state = searchService.get_context_features(state)
        recent_context_df = searchService.get_recent_context(current_state)
        # context_df = searchService.get_context_df(current_state, recent_context_df)
        return recent_context_df

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = '', context = pd.DataFrame) -> str:
        """Wraps the standard adaptive explanation prompt."""
        input_dict = json.loads(x)
        # context_df = AdaXTask.prepare_context(input_dict)
        return standard_prompt.format(
            # task_description=input_dict["task_description"],
            context=context,
            physiological_state=input_dict["physiological_state"],
            behavioral_state=input_dict["behavioral_state"]
            ) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '', context = pd.DataFrame) -> str:
        """Wraps the Chain-of-Thought (CoT) adaptive explanation prompt."""
        input_dict = json.loads(x)
        # context_df = AdaXTask.prepare_context(input_dict)
        
        return cot_prompt.format(
            # task_description=input_dict["task_description"],
            context=context,
            physiological_state=input_dict["physiological_state"],
            behavioral_state=input_dict["behavioral_state"]
            ) + y

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        """Wraps the voting prompt for selecting the most adaptive explanation."""
        prompt = vote_prompt
        for i, y in enumerate(ys, 1):
            try:
                # Attempt to parse y as JSON for proper formatting
                parsed_y = json.loads(y) if isinstance(y, str) else y
                formatted_y = json.dumps(parsed_y, indent=4)  # Pretty-print JSON
            except json.JSONDecodeError:
                formatted_y = str(y)  # Fallback to string if JSON parsing fails
        
            prompt += f'Choice {i}:\n{formatted_y}\n'
        return prompt

    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        """Extracts voting results to determine the best adaptive explanation."""
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best adaptive explanation is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        """Wraps the comparison prompt to refine explanations."""
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        ys = [y.split('Explanation:\n')[-1] for y in ys]
        prompt = compare_prompt + f'Explanation 1:\n{ys[0]}\n\nExplanation 2:\n{ys[1]}\n'
        return prompt

    @staticmethod
    def compare_output_unwrap(compare_output: str):
        """Extracts comparison result to determine the best explanation adjustment."""
        if 'more coherent passage is 1' in compare_output:
            return 0
        elif 'more coherent passage is 2' in compare_output:
            return 1
        elif 'two passages are similarly coherent' in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1
