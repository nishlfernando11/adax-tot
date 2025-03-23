import argparse
from tot.methods.bfs import solve
from tot.methods.standard import genAdaX
from tot.tasks.adax import AdaXTask
import time

args = argparse.Namespace(backend='mistral:latest', temperature=0.6, task='adax', naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1)
# args = argparse.Namespace(backend='openai', temperature=0.7, task='adax', naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=3, n_evaluate_sample=3, n_select_sample=1)

task = AdaXTask()
# ys, infos = solve(args, task, 0)
for idx in range(4):
    start = time.time()
    print(f"Running {idx}")
    ys = genAdaX(args, task, idx)
    print(f"Time: {time.time() - start}")
    print(ys)
