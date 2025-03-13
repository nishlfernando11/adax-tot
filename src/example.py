import argparse
from tot.methods.bfs import solve
from tot.tasks.adax import AdaXTask


# args = argparse.Namespace(backend='gpt-4', temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

# task = Game24Task()
# ys, infos = solve(args, task, 900)
# print(ys[0])

# args = argparse.Namespace(backend='gpt-4', temperature=0.8, task='adax', naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=5, n_evaluate_sample=5, n_select_sample=1)
args = argparse.Namespace(backend='mistral:latest', temperature=0.7, task='adax', naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=3, n_evaluate_sample=3, n_select_sample=1)

task = AdaXTask()
ys, infos = solve(args, task, 0)
print(ys[0])


# args = argparse.Namespace(backend='gpt-4', temperature=1.0, task='text', naive_run=False, prompt_sample="cot", method_generate='sample', method_evaluate='vote', method_select='greedy', n_generate_sample=5, n_evaluate_sample=5, n_select_sample=1)

# task = TextTask()
# ys, infos = solve(args, task, 0)
# print(ys[0])