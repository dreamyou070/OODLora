import torch, ast


def arg_as_list(arg):
    v = ast.literal_eval(arg)
    return v

arg = [32,64]
print(arg_as_list(arg))