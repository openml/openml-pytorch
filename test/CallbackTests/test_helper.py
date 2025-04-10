from typing import Iterable
import unittest
import torch
from unittest.mock import MagicMock
from openml_pytorch.callbacks.helper import listify, camel2snake, _camel_re1, _camel_re2
from functools import partial


class TestHelper(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.cases = [(None, []), ([1, 2], [1, 2]), ("st", ["st"]), ((1, 2), [1, 2])]

    def test_listify(self):
        for input_val, expected in self.cases:
            with self.subTest(input=input_val):
                self.assertEqual(listify(input_val), expected)


class TestCamel2snake(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.cases = [("HelloWorld", "hello_world"), ("", ""), ("hiWorld", "hi_world")]

    def test_camel2snake(self):
        for input_val, expected in self.cases:
            with self.subTest(input=input_val):
                self.assertEqual(camel2snake(input_val), expected)
