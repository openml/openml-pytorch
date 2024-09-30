# Callbacks
Callbacks module contains classes and functions for handling callback functions during an event-driven process. This makes it easier to customize the behavior of the training loop and add additional functionality to the training process without modifying the core code.

To use a callback, create a class that inherits from the Callback class and implement the necessary methods. Callbacks can be used to perform actions at different stages of the training process, such as at the beginning or end of an epoch, batch, or fitting process. Then pass the callback object to the Trainer.

::: callbacks