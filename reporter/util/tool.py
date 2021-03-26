import math
import torch
from typing import Generator, Iterable, TypeVar

A = TypeVar('A')


def takeuntil(target: A, xs: Iterable[A]) -> Generator[A, None, A]:
    '''
    >>> list(takeuntil('</s>', ['I', 'am', 'Kirito', '</s>', '</s>', '</s>']))
    ['I', 'am', 'Kirito', '</s>']
    '''
    for x in xs:
        yield x
        if x == target:
            return

def log1mexp(x, expm1_guard = 1e-7):
    # (6) in https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    assert(all(x < .0))

    t = x < math.log(0.5)
    y = torch.zeros_like(x)
    y[t] = torch.log1p(-x[t].exp())

    # for x close to 0 we need expm1 for numerically stable computation
    # we furtmermore modify the backward pass to avoid instable gradients,
    # ie situations where the incoming output gradient is close to 0 and the gradient of expm1 is very large
    expxm1 = torch.expm1(x[~t])
    log1mexp_fw = (-expxm1).log()
    log1mexp_bw = (-expxm1+expm1_guard).log() # limits magnitude of gradient

    y[~t] = log1mexp_fw.detach() + (log1mexp_bw - log1mexp_bw.detach())
    return y

