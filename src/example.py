import argparse
from tot.methods.bfs import solve
from tot.methods.standard import genAdaX
from tot.tasks.adax import AdaXTask
from tot.models import gpt_usage
import time
import json
# args = argparse.Namespace(backend='mistral:7b-instruct-q4_0', temperature=0.6, task='adax', naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=3, n_evaluate_sample=3, n_select_sample=1)
args = argparse.Namespace(backend='gpt-3.5-turbo', temperature=0.6, task='adax', is_local=False, naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=3, n_evaluate_sample=3, n_select_sample=1, stop=None)

task = AdaXTask(isLive=False)
ys, infos = solve(args, task, 0)
print(ys)
json_ys = json.loads(ys[0])
print(json_ys)
print('usage_so_far', gpt_usage(args.backend))

# for idx in range(6):
#     start = time.time()
#     print(f"Running {idx}")
#     ys = genAdaX(args, task, idx)
#     print(f"Time: {time.time() - start}")
#     print(ys)