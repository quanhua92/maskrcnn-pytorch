import torch as th


def no_grad(f):
    def new_f(*args, **kwargs):
        with th.no_grad():
            return f(*args, **kwargs)
    return new_f
