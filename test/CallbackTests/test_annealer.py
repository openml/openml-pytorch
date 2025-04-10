import unittest
import torch
from unittest.mock import MagicMock
from openml_pytorch.callbacks.annealing import sched_cos, sched_lin, sched_no, sched_exp
from functools import partial

class TestAnnealer(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.start = 0.0
        self.end = 1.0

    def test_sched_lin(self):
        pos = 0.6
        expected = 0.6
        scheduler = sched_lin(self.start, self.end)
        assert expected == scheduler(pos)
    
    def test_sched_cos(self):
        pos = 0.6
        expected = 0.6545084971874737
        scheduler = sched_cos(self.start, self.end)
        assert expected == scheduler(pos)

    def test_sched_no(self):
        pos = 0.6
        expected = self.start
        scheduler = sched_no(self.start, self.end)
        assert expected == scheduler(pos)

    def test_sched_exp(self):
        pos = 0.6
        expected = 0.009999999999999997
        scheduler = sched_exp(self.start, self.end)
        assert expected == scheduler(pos) 
