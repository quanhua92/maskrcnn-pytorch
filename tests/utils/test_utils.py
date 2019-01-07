import torch as th
from maskrcnn.lib.utils.torch_utils import no_grad


@no_grad
def f(a):
    return a * 1


def test_no_grad():
    a = th.zeros(1, requires_grad=True)

    assert a.requires_grad is True

    output = f(a)

    assert output.requires_grad is False



