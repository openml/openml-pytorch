from .helper import listify
from typing import Iterable
import math
from functools import partial
import torch
from .callback import Callback


def annealer(f) -> callable:
    """
    A decorator function for creating a partially applied function with predefined start and end arguments.
    The inner function `_inner` captures the `start` and `end` parameters and returns a `partial` object that fixes these parameters for the decorated function `f`.
    """

    def _inner(start, end):
        return partial(f, start, end)

    return _inner


@annealer
def sched_lin(start: float, end: float, pos: float) -> float:
    """
    A linear schedule function.
    """
    return start + pos * (end - start)


@annealer
def sched_cos(start: float, end: float, pos: float) -> float:
    """
    A cosine schedule function.
    """
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def sched_no(start: float, end: float, pos: float) -> float:
    """
    Disabled scheduling.
    """
    return start


@annealer
def sched_exp(start: float, end: float, pos: float) -> float:
    """
    Exponential schedule function.
    """
    return start * (end / start) ** pos


def combine_scheds(pcts: Iterable[float], scheds: Iterable[callable]) -> callable:
    """
    Combine multiple scheduling functions.
    """
    assert sum(pcts) == 1.0
    pcts = torch.tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos)
    
    return _inner

class ParamScheduler(Callback):
    """
    Manages scheduling of parameter adjustments over the course of training.
    """

    _order = 1

    def __init__(self, pname, sched_funcs):
        self.pname, self.sched_funcs = pname, sched_funcs

    def begin_fit(self):
        """
        Prepare the scheduler at the start of the fitting process.
        This method ensures that sched_funcs is a list with one function per parameter group.
        """
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        """
        Adjust the parameter value for each parameter group based on the scheduling function.
        Ensures the number of scheduling functions matches the number of parameter groups.
        """
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = f(self.n_epochs / self.epochs)

    def begin_batch(self):
        """
        Apply parameter adjustments at the beginning of each batch if in training mode.
        """
        if self.in_train:
            self.set_param()