import os
import openai
import backoff 
import ollama
from openai import OpenAI
from dotenv import load_dotenv
from tot.prompts.adax import STATIC_KNOWLEDGE_BASE
load_dotenv(dotenv_path="./.env")
completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")

api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base
    
client = OpenAI()

@backoff.on_exception(backoff.expo, openai.OpenAIError) #DEV: updated openai.error.OpenAIError to openai.OpenAIError
def completions_with_backoff(**kwargs):
    # return openai.ChatCompletion.create(**kwargs)
    return client.chat.completions.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1100, n=1, stop=None, parallel=False) -> list:
    messages = [{
                        "role": "system",
                        "content": STATIC_KNOWLEDGE_BASE
                    },
                {"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop, parallel=parallel)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None,parallel=False) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    
    if not parallel:
        # Original sequential batching logic
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            res = completions_with_backoff(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=cnt,
                stop=stop
            )
            outputs.extend([choice.message.content for choice in res.choices])
            completion_tokens += res.usage.completion_tokens
            prompt_tokens += res.usage.prompt_tokens
        return outputs

    # Parallel execution
    def call_once():
        try:
            res = completions_with_backoff(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=stop
            )
            return res.choices[0].message.content, res.usage.completion_tokens, res.usage.prompt_tokens
        except Exception as e:
            print(f"Error during parallel call: {e}")
            return "Error", 0, 0

    with ThreadPoolExecutor(max_workers=min(n, 10)) as executor:
        futures = [executor.submit(call_once) for _ in range(n)]
        for future in as_completed(futures):
            output, c_tokens, p_tokens = future.result()
            outputs.append(output)
            completion_tokens += c_tokens
            prompt_tokens += p_tokens

    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

#---------#
# Ollama  #
#---------#

import os
import subprocess

# STATIC_KNOWLEDGE_BASE = """
# # Role and Purpose:
# You are an AI assistant specializing in adaptive explainability for Human-Machine Teaming (HMT).
# Your task is to generate concise, 10-word explanations for AI behavior in Overcooked AI.

# # Constraints:
# - Explanations must adapt to the user’s current stress, trust, and cognitive load.
# - Adapt phrasing and focus using game metrics (score, collisions, time).
# - Keep the explanation simple but meaningful — under pressure. Maintain clarity under pressure.
# - Use adaptive explainability features: [duration: (short/long), granularity: (highlevel/detailed), timing: (reactive, proactive)].
# - Reason only about the AI agent’s decisions and behaviors.
# - When needed, include corrective or supportive tone based on user signals.
# - Each explanation must include adaptive features used to generate that explanation: [duration, granularity, timing].
# - Justify explanation type chosen.
# """

completion_tokens = prompt_tokens = 0  # Tracking tokens (if needed)

def local_model(prompt, model="mistral:7b-instruct-q4_K_M", temperature=0.7, max_tokens=300, n=1, stop=None) -> list:
    # Set max_tokens to 100 from 1000
    """
    Runs Mistral locally using Ollama instead of OpenAI's API.
    
    :param prompt: User input prompt
    :param model: Ollama model name (default: mistral)
    :param temperature: Sampling temperature
    :param max_tokens: Maximum output tokens
    :param n: Number of completions to generate
    :param stop: Stop sequence for termination
    :return: List of generated responses
    """
    messages = [{"role": "user", "content": prompt}]
    return ollama_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

"""
Ollama chat with ollama run command
"""
def ollama_chat_process(messages, model="mistral:7b-instruct-q4_K_M", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    """
    Calls Ollama's locally running model to process chat messages.
    
    :param messages: List of messages [{role: "user", content: "..."}]
    :param model: Ollama model name (default: mistral)
    :param temperature: Sampling temperature
    :param max_tokens: Maximum response length
    :param n: Number of completions
    :param stop: Stop sequences
    :return: List of responses
    """
    global completion_tokens, prompt_tokens
    outputs = []

    for _ in range(n):  # Generate 'n' responses
        # Construct prompt as a single string (Ollama doesn't support message history directly)
        prompt_text = messages[-1]["content"]  # Use only the last message

        try:
            # Run Ollama with local Mistral model
            result = subprocess.run(
                ["ollama", "run", model, prompt_text],
                capture_output=True,
                text=True
            )
            output_text = result.stdout.strip()
            outputs.append(output_text)

            # Simple token estimation
            completion_tokens += len(output_text.split())
            prompt_tokens += len(prompt_text.split())

        except Exception as e:
            print(f"Error running local Mistral: {e}")
            outputs.append("Error")

    return outputs


# def ollama_chat(messages, model="mistral:latest", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
#     """
#     Calls the locally running Ollama model using the Ollama Python library.
    
#     :param messages: List of messages [{role: "user", content: "..."}]
#     :param model: Ollama model name (default: mistral)
#     :param temperature: Sampling temperature
#     :param max_tokens: Maximum response length
#     :param n: Number of completions
#     :param stop: Stop sequences
#     :return: List of responses
#     """
#     global completion_tokens, prompt_tokens
#     outputs = []

#     for _ in range(n):  # Generate 'n' responses
#         prompt_text = messages[-1]["content"]

#         try:
#             # Call Ollama using its Python library
#             response = ollama.chat(
#                 model=model,
#                 messages=[{"role": "user", "content": prompt_text}],
#                 options={"temperature": temperature, "num_predict": max_tokens}
#             )
#             output_text = response["message"]["content"]
#             outputs.append(output_text)

#             # Simple token estimation
#             completion_tokens += len(output_text.split())
#             prompt_tokens += len(prompt_text.split())

#         except Exception as e:
#             print(f"Error running local Mistral: {e}")
#             outputs.append("Error")

#     return outputs


from concurrent.futures import ThreadPoolExecutor, as_completed

def ollama_chat(messages, model="mistral:7b-instruct-q4_K_M", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    """
    Calls the locally running Ollama model using the Ollama Python library in parallel.
    
    :param messages: List of messages [{role: "user", content: "..."}]
    :param model: Ollama model name (default: mistral:7b-instruct-q4_K_M)
    :param temperature: Sampling temperature
    :param max_tokens: Maximum response length
    :param n: Number of completions
    :param stop: Stop sequences
    :return: List of responses
    """
    global completion_tokens, prompt_tokens
    outputs = []
    completion_tokens = 0
    prompt_text = messages[-1]["content"] 
    prompt_tokens = len(prompt_text.split())  # Compute prompt tokens once
    

    def call_ollama():
        """Calls the local Ollama model and returns the response."""
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": STATIC_KNOWLEDGE_BASE
                    },
                    {"role": "user", "content": prompt_text}],
                options={"temperature": temperature, "num_predict": max_tokens}
            )
            output_text = response["message"]["content"]
            return output_text, len(output_text.split())  # Return response and token count
        except Exception as e:
            print(f"Error running local Mistral: {e}")
            return "Error", 0

    # Run API calls in parallel (max 5 concurrent requests)
    with ThreadPoolExecutor(max_workers=min(n, 5)) as executor:
        futures = [executor.submit(call_ollama) for _ in range(n)]
        
        for future in as_completed(futures):
            output_text, tokens = future.result()
            outputs.append(output_text)
            completion_tokens += tokens  # Accumulate completion tokens

    return outputs


def local_usage():
    """
    Tracks estimated token usage (Ollama does not return token usage).
    
    :return: Dictionary with token usage data
    """
    global completion_tokens, prompt_tokens
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens
    }
    
    

