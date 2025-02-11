import re
from .helper import camel2snake

class Callback:
    """

    Callback class is a base class designed for handling different callback functions during
    an event-driven process. It provides functionality to set a runner, retrieve the class
    name in snake_case format, directly call callback methods, and delegate attribute access
    to the runner if the attribute does not exist in the Callback class.

    The _order is used to decide the order of Callbacks.

    """

    _order = 0

    def set_runner(self, run) -> None:
        self.run = run

    @property
    def name(self):
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return camel2snake(name or "callback")

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False

    def __getattr__(self, k):
        return getattr(self.run, k)