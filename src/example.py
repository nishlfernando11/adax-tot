import argparse
from tot.methods.bfs import solve
from tot.tasks.adax import AdaXTask

args = argparse.Namespace(backend='mistral:latest', temperature=0.7, task='adax', naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=3, n_evaluate_sample=3, n_select_sample=1)

task = AdaXTask()
ys, infos = solve(args, task, 0)
print(ys[0])
