"""
This module provides classes and methods to facilitate the configuration, data handling, training, and evaluation of machine learning models using PyTorch and OpenML datasets. The functionalities include:
- Generation of default configurations for models.
- Handling of image and tabular data.
- Training and evaluating machine learning models.
- Exporting trained models to ONNX format.
- Managing data transformations and loaders.
"""

import copy
from datetime import datetime
import gc
import io
import logging
import warnings
from collections import OrderedDict
from functools import partial
from types import SimpleNamespace
from typing import Any, List, Optional

import netron
import numpy as np
import onnx
import pandas as pd
import torch
import torch.amp
import torch.utils
from openml.exceptions import PyOpenMLError
from openml.tasks import (
    OpenMLClassificationTask,
    OpenMLRegressionTask,
    OpenMLSupervisedTask,
    OpenMLTask,
)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Resize, ToPILImage, ToTensor
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from .callbacks import *
from .custom_datasets import OpenMLImageDataset, OpenMLTabularDataset
from .metrics import accuracy


def convert_to_rgb(image):
    """
    Converts an image to RGB mode if it is not already in that mode.

    Parameters:
    image (PIL.Image): The image to be converted.

    Returns:
    PIL.Image: The converted image in RGB mode.
    """
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


class DefaultConfigGenerator:
    """
    DefaultConfigGenerator class provides various methods to generate default configurations.
    """

    @staticmethod
    def _default_criterion_gen(task: OpenMLTask) -> torch.nn.Module:
        """
        _default_criterion_gen returns a criterion based on the task type - regressions use
        torch.nn.SmoothL1Loss while classifications use torch.nn.CrossEntropyLoss
        """
        if isinstance(task, OpenMLRegressionTask):
            return torch.nn.SmoothL1Loss()
        elif isinstance(task, OpenMLClassificationTask):
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(task)

    @staticmethod
    def _default_optimizer_gen(model: torch.nn.Module, _: OpenMLTask):
        """
        _default_optimizer_gen returns the torch.optim.Adam optimizer for the given model
        """
        return torch.optim.Adam

    @staticmethod
    def _default_scheduler_gen(optim, _: OpenMLTask) -> Any:
        """
        _default_scheduler_gen returns the torch.optim.lr_scheduler.ReduceLROnPlateau scheduler for the given optimizer
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim)

    @staticmethod
    def _default_predict(output: torch.Tensor, task: OpenMLTask) -> torch.Tensor:
        """
        _default_predict turns the outputs into predictions by returning the argmax of the output tensor for classification tasks, and by flattening the prediction in case of the regression
        """
        output_axis = output.dim() - 1
        if isinstance(task, OpenMLClassificationTask):
            output = torch.argmax(output, dim=output_axis)
        elif isinstance(task, OpenMLRegressionTask):
            output = output.view(-1)
        else:
            raise ValueError(task)
        return output

    @staticmethod
    def _default_predict_proba(output: torch.Tensor, task: OpenMLTask) -> torch.Tensor:
        """
        _default_predict_proba turns the outputs into probabilities using softmax
        """
        output_axis = output.dim() - 1
        output = output.softmax(dim=output_axis)
        return output

    @staticmethod
    def _default_sanitize(tensor: torch.Tensor) -> torch.Tensor:
        """
        _default sanitizer replaces NaNs with 1e-6
        """
        tensor = torch.where(
            torch.isnan(tensor), torch.ones_like(tensor) * torch.tensor(1e-6), tensor
        )
        return tensor

    @staticmethod
    def _default_retype_labels(tensor: torch.Tensor, task: OpenMLTask) -> torch.Tensor:
        """
        _default_retype_labels changes the type of the tensor to long for classification tasks and to float for regression tasks
        """
        if isinstance(task, OpenMLClassificationTask):
            return tensor.long()
        elif isinstance(task, OpenMLRegressionTask):
            return tensor.float()
        else:
            raise ValueError(task)

    def get_device(
        self,
    ):
        """
        Checks if a GPU is available and returns the device to be used for training (cuda, mps or cpu)
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        return device

    def default_image_transform(self):
        return Compose(
            [
                ToPILImage(),  # Convert tensor to PIL Image to ensure PIL Image operations can be applied.
                Lambda(convert_to_rgb),  # Convert PIL Image to RGB if it's not already.
                Resize((128, 128)),  # Resize the image.
                ToTensor(),  # Convert the PIL Image back to a tensor.
            ]
        )

    def return_model_config(self):
        """
        Returns a configuration object for the model
        """

        return SimpleNamespace(
            device=self.get_device(),
            criterion=self._default_criterion_gen,
            optimizer_gen=self._default_optimizer_gen,
            scheduler_gen=self._default_scheduler_gen,
            # predict turns the outputs of the model into actual predictions
            predict=self._default_predict,  # type: Callable[[torch.Tensor, OpenMLTask], torch.Tensor]
            # predict_proba turns the outputs of the model into probabilities for each class
            predict_proba=self._default_predict_proba,  # type: Callable[[torch.Tensor], torch.Tensor]
            # epoch_count represents the number of epochs the model should be trained for
            epoch_count=3,  # type: int,
            # progress_callback=(
            #     self._default_progress_callback
            # ),  # type: Callable[[int, int, int, int, float, float], None]
            # enable progress bar
            verbose=True,
        )

    def return_data_config(self):
        """
        Returns a configuration object for the data
        """
        return SimpleNamespace(
            type_of_data="image",
            perform_validation=False,
            # progress_callback is called when a training step is finished, in order to report the current progress
            # sanitize sanitizes the input data in order to ensure that models can be trained safely
            sanitize=self._default_sanitize,  # type: Callable[[torch.Tensor], torch.Tensor]
            # retype_labels changes the types of the labels in order to ensure type compatibility
            retype_labels=(self._default_retype_labels),  # type: Callable[[torch.Tensor, OpenMLTask], torch.Tensor]
            # image_size is the size of the images that are fed into the model
            image_size=128,
            # batch_size represents the processing batch size for training
            batch_size=64,  # type: int
            data_augmentation=None,
            validation_split=0.1,
            transform=self.default_image_transform(),
        )


class BaseDataHandler:
    """
    BaseDataHandler class is an abstract base class for data handling operations.
    """

    def prepare_data(
        self, X_train, y_train, X_val, y_val, data_config: SimpleNamespace
    ):
        raise NotImplementedError

    def prepare_test_data(self, X_test, data_config: SimpleNamespace):
        raise NotImplementedError


class OpenMLImageHandler(BaseDataHandler):
    """
    OpenMLImageHandler is a class that extends BaseDataHandler to handle image data from OpenML datasets.
    """

    def prepare_data(self, X_train, y_train, X_val, y_val, data_config=None):
        train = OpenMLImageDataset(
            image_dir=data_config.file_dir,
            X=X_train,
            y=y_train,
            transform_x=data_config.transform,
            image_size=data_config.image_size,
        )
        val = OpenMLImageDataset(
            image_dir=data_config.file_dir,
            X=X_val,
            y=y_val,
            transform_x=data_config.transform,
            image_size=data_config.image_size,
        )
        return train, val

    def prepare_test_data(self, X_test, data_config=None):
        test = OpenMLImageDataset(
            image_dir=data_config.file_dir,
            X=X_test,
            y=None,
            transform_x=data_config.transform,
            image_size=data_config.image_size,
        )
        return test


class OpenMLTabularHandler(BaseDataHandler):
    """
    OpenMLTabularHandler is a class that extends BaseDataHandler to handle tabular data from OpenML datasets.
    """

    def prepare_data(self, X_train, y_train, X_val, y_val, data_config=None):
        train = OpenMLTabularDataset(X=X_train, y=y_train)
        val = OpenMLTabularDataset(X=X_val, y=y_val)
        return train, val

    def prepare_test_data(self, X_test, data_config=None):
        test = OpenMLTabularDataset(X=X_test, y=None)
        return test


# Dictionary mapping data types to handlers
data_handlers = {
    "image": OpenMLImageHandler(),
    "dataframe": OpenMLTabularHandler(),
    # Add new data types here
}


class DataContainer:
    """
    class DataContainer:
        A class to contain the training, validation, and test data loaders. This just makes it easier to access them when required.

        Attributes:
        train_dl: DataLoader object for the training data.
        valid_dl: DataLoader object for the validation data.
        test_dl: Optional DataLoader object for the test data.
    """

    def __init__(
        self,
        train_dl: DataLoader,
        valid_dl: torch.utils.data.DataLoader,
        test_dl: torch.utils.data.DataLoader = None,
    ):  # type: ignore
        self.train_dl, self.valid_dl = train_dl, valid_dl
        self.test_dl = test_dl

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset

    @property
    def test_ds(self):
        return self.test_dl.dataset


class OpenMLDataModule:
    def __init__(
        self,
        type_of_data="image",
        filename_col="Filename",
        file_dir="images",
        target_mode="categorical",
        transform=None,
        target_column="encoded_labels",
        **kwargs,
    ):
        self.config_gen = DefaultConfigGenerator()
        self.data_config = self.config_gen.return_data_config()
        self.data_config.type_of_data = type_of_data
        self.data_config.filename_col = filename_col
        self.data_config.file_dir = file_dir
        self.data_config.target_mode = target_mode
        self.data_config.target_column = target_column
        self.handler: BaseDataHandler | None = data_handlers.get(type_of_data)

        if transform is not None:
            self.data_config.transform = transform

        if not self.handler:
            raise ValueError(f"Data type {type_of_data} not supported.")

    def get_data(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series],
        X_test: pd.DataFrame,
        task,
    ):
        # Split the training data
        X_train_train, X_val, y_train_train, y_val = self.split_training_data(X_train, y_train)

        y_train_train, y_val, model_classes = self.encode_labels(y_train_train, y_val)

        # Use handler to prepare datasets
        train_loader, val_loader = self.prepare_datasets_for_training_and_validation(X_train_train, X_val, y_train_train, y_val)

        # Prepare test data
        test_loader = self.process_test_data(X_test)

        return DataContainer(train_loader, val_loader, test_loader), model_classes

    def process_test_data(self, X_test):
        test = self.handler.prepare_test_data(X_test, self.data_config)
        test_loader = DataLoader(
            test, batch_size=self.data_config.batch_size, shuffle=False
        )
        
        return test_loader

    def prepare_datasets_for_training_and_validation(self, X_train_train, X_val, y_train_train, y_val):
        if self.handler is None:
            raise ValueError(
                f"Data type {self.data_config.type_of_data} not supported."
            )

        train, val = self.handler.prepare_data(
            X_train_train, y_train_train, X_val, y_val, self.data_config
        )

        train_loader = DataLoader(
            train, batch_size=self.data_config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val, batch_size=self.data_config.batch_size, shuffle=False
        )
        
        return train_loader,val_loader

    def encode_labels(self, y_train_train, y_val):
        """
        Encode the labels for categorical data
        """
        if self.data_config.target_mode == "categorical":
            label_encoder = preprocessing.LabelEncoder().fit(y_train_train)
            y_train_train = pd.Series(label_encoder.transform(y_train_train))
            y_val = pd.Series(label_encoder.transform(y_val))
            # Determine model classes
            model_classes = (
                label_encoder.classes_
                if self.data_config.target_mode == "categorical"
                else None
            )
            
        return y_train_train,y_val,model_classes

    def split_training_data(self, X_train, y_train):
        if type(y_train) != pd.Series:
            y_train = pd.Series(y_train)

        X_train_train, X_val, y_train_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.data_config.validation_split,
            shuffle=True,
            stratify=y_train,
            random_state=0,
        )
        
        return X_train_train,X_val,y_train_train,y_val


class ModelRunner:
    def __init__(self, cbs=None, cb_funcs=None):
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop, self.cbs = False, [TrainEvalCallback()] + cbs

    @property
    def opt(self):
        return self.learn.opt

    @property
    def model(self):
        return self.learn.model

    @property
    def criterion(self):
        return self.learn.criterion

    @property
    def data(self):
        return self.learn.data

    @property
    def label_mapping(self):
        return self.learn.label_mapping

    @property
    def model_classes(self):
        return self.learn.model_classes

    def one_batch(self, xb, yb):
        try:
            self.xb, self.yb = xb, yb

            self("begin_batch")
            # Below two lines are hack to convert model to onnx
            global sample_input
            sample_input = self.xb
            self.pred = self.model(self.xb)
            self("after_pred")
            self.loss = self.criterion(self.pred, self.yb)
            self("after_loss")
            if not self.in_train:
                return
            self.loss.backward()
            self("after_backward")
            self.opt.step()
            self("after_step")
            self.opt.zero_grad()
        except CancelBatchException:
            self("after_cancel_batch")
        finally:
            self("after_batch")

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb, yb in tqdm(dl, leave=False):
                self.one_batch(xb, yb)
        except CancelEpochException:
            self("after_cancel_epoch")

    def fit(self, epochs, learn):
        self.epochs, self.learn, self.loss = epochs, learn, torch.tensor(0.0)
        try:
            for cb in self.cbs:
                cb.set_runner(self)
            self("begin_fit")
            for epoch in range(epochs):
                self.epoch = epoch
                if not self("begin_epoch"):
                    self.all_batches(self.data.train_dl)
                with torch.no_grad():
                    if not self("begin_validate"):
                        self.all_batches(self.data.valid_dl)
                self("after_epoch")
        except CancelTrainException:
            self("after_cancel_train")
        finally:
            self("after_fit")
            self.learn = None

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res


class Learner:
    """
    A class to store the model, optimizer, criterion, and data loaders for training and evaluation.
    """

    def __init__(self, model, opt, criterion, data, model_classes, device="cpu"):
        (
            self.model,
            self.opt,
            self.criterion,
            self.data,
            self.model_classes,
            self.device,
        ) = (model, opt, criterion, data, model_classes, device)


class OpenMLTrainerModule:
    def _default_progress_callback(
        self, fold: int, rep: int, epoch: int, step: int, loss: float, accuracy: float
    ):
        # todo : move this into callback
        """
                _default_progress_callback reports the current fold, rep, epoch, step and loss for every
        training iteration to the default logger
        """
        self.logger.info(
            "[%d, %d, %d, %d] loss: %.4f, accuracy: %.4f"
            % (fold, rep, epoch, step, loss, accuracy)
        )

    def __init__(
        self,
        experiment_name: str,
        data_module: OpenMLDataModule,
        callbacks: List[Callback] = [],
        use_tensorboard: bool = True,
        **kwargs,
    ):
        self.experiment_name = experiment_name
        self.config_gen = DefaultConfigGenerator()
        self.model_config = self.config_gen.return_model_config()
        self.data_module = data_module
        self.callbacks = callbacks

        self.config = SimpleNamespace(
            **{**self.model_config.__dict__, **self.data_module.data_config.__dict__}
        )
        # update the config with the user defined values
        self.config.__dict__.update(kwargs)
        self.config.progress_callback = self._default_progress_callback
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.user_defined_measures = OrderedDict()
        # Tensorboard support
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.tensorboard_writer = None

        if use_tensorboard:
            self.tensorboard_writer = SummaryWriter(
                comment=experiment_name,
                log_dir=f"tensorboard_logs/{experiment_name}/{timestamp}",
            )

        # self.callbacks.append(LoggingCallback(self.logger, print_output=False))
        self.loss = 0
        self.training_state = True

        self.phases = [0.2, 0.8]
        self.scheds = combine_scheds(
            self.phases, [sched_cos(1e-4, 1e-2), sched_cos(1e-3, 1e-5)]
        )

        # Add default callbacks
        self.cbfs = [
            Recorder,
            partial(AvgStatsCallback, accuracy),
            partial(ParamScheduler, "lr", self.scheds),
            partial(PutDataOnDeviceCallback, self.config.device),
        ]
        if self.tensorboard_writer is not None:
            self.cbfs.append(partial(TensorBoardCallback, self.tensorboard_writer))

        self.add_callbacks()

    def export_to_onnx(self, model_copy):
        """
        Converts the model to ONNX format. Uses a hack for now (global variable) to get the sample input.
        """
        global sample_input
        f = io.BytesIO()
        torch.onnx.export(model_copy, sample_input, f)
        onnx_model = onnx.load_model_from_string(f.getvalue())
        onnx_ = onnx_model.SerializeToString()
        global last_models
        last_models = onnx_
        return onnx_

    def export_to_netron(self, onnx_file_name: str = f"model.onnx"):
        """
        Exports the model to ONNX format and serves it using netron.
        """
        if self.onnx_model is None:
            try:
                self.onnx_model = self.export_to_onnx(self.model)
            except Exception as e:
                raise ValueError("Model is not defined")
            raise ValueError("Model is not defined")

        # write the onnx model to a file
        with open(onnx_file_name, "wb") as f:
            f.write(self.onnx_model)
            print(f"Writing onnx model to {onnx_file_name}. Delete if neeeded")

        # serve with netro
        netron.start(onnx_file_name)

    def run_model_on_fold(
        self,
        model: torch.nn.Module,
        task: OpenMLTask,
        X_train: pd.DataFrame,
        rep_no: int,
        fold_no: int,
        y_train: Optional[pd.Series],
        X_test: pd.DataFrame,
    ):
        # if task has no class labels, we assign the class labels to be the unique values in the training set
        if task.class_labels is None:
            task.class_labels = y_train.unique()

        # Add the user defined callbacks

        self.model = copy.deepcopy(model)

        try:
            data, model_classes = self.run_training(task, X_train, y_train, X_test)

        except AttributeError as e:
            # typically happens when training a regressor8 on classification task
            raise PyOpenMLError(str(e))

        # In supervised learning this returns the predictions for Y
        pred_y, proba_y = self.run_evaluation(task, data, model_classes)

        # Convert predictions to class labels
        if task.class_labels is not None:
            pred_y = [task.class_labels[i] for i in pred_y]

        # Convert model to onnx
        onnx_ = self.export_to_onnx(self.model)

        # Hack to store the last model for ONNX conversion
        global last_models
        # last_models = onnx_
        self.onnx_model = onnx_

        return pred_y, proba_y, self.user_defined_measures, None

    def check_config(self):
        raise NotImplementedError

    def _prediction_to_probabilities(
        self, y: np.ndarray, classes: List[Any]
    ) -> np.ndarray:
        """Transforms predicted probabilities to match with OpenML class indices.

        Parameters
        ----------
        y : np.ndarray
            Predicted probabilities (possibly omitting classes if they were not present in the
            training data).
        model_classes : list
            List of classes known_predicted by the model, ordered by their index.

        Returns
        -------
        np.ndarray
        """
        # y: list or numpy array of predictions
        # model_classes: mapping from original array id to
        # prediction index id
        if not isinstance(classes, list):
            raise ValueError(
                "please convert model classes to list prior to calling this fn"
            )
        result = np.zeros((len(y), len(classes)), dtype=np.float32)
        for obs, prediction_idx in enumerate(y):
            result[obs][prediction_idx] = 1.0
        return result

    def run_evaluation(self, task, data, model_classes):
        if isinstance(task, OpenMLSupervisedTask):
            self.model.eval()
            pred_y = self.pred_test(task, self.model, data.test_dl, self.config.predict)
        else:
            raise ValueError(task)

        if isinstance(task, OpenMLClassificationTask):
            try:
                self.model.eval()
                proba_y = self.pred_test(
                    task, self.model, data.test_dl, self.config.predict_proba
                )

            except AttributeError:
                if task.class_labels is not None:
                    proba_y = self._prediction_to_probabilities(
                        pred_y, list(task.class_labels)
                    )
                else:
                    raise ValueError("The task has no class labels")

            if task.class_labels is not None:
                if proba_y.shape[1] != len(task.class_labels):
                    # Remap the probabilities in case there was a class missing
                    # at training time. By default, the classification targets
                    # are mapped to be zero-based indices to the actual classes.
                    # Therefore, the model_classes contain the correct indices to
                    # the correct probability array. Example:
                    # classes in the dataset: 0, 1, 2, 3, 4, 5
                    # classes in the training set: 0, 1, 2, 4, 5
                    # then we need to add a column full of zeros into the probabilities
                    # for class 3 because the rest of the library expects that the
                    # probabilities are ordered the same way as the classes are ordered).
                    proba_y_new = np.zeros((proba_y.shape[0], len(task.class_labels)))
                    for idx, model_class in enumerate(model_classes):
                        proba_y_new[:, model_class] = proba_y[:, idx]
                    proba_y = proba_y_new

                if proba_y.shape[1] != len(task.class_labels):
                    message = "Estimator only predicted for {}/{} classes!".format(
                        proba_y.shape[1],
                        len(task.class_labels),
                    )
                    warnings.warn(message)
                    self.logger.warning(message)
            else:
                raise ValueError("The task has no class labels")

        elif isinstance(task, OpenMLRegressionTask):
            proba_y = None

        else:
            raise TypeError(type(task))
        return pred_y, proba_y

    def run_training(self, task, X_train, y_train, X_test):
        if isinstance(task, OpenMLSupervisedTask) or isinstance(
            task, OpenMLClassificationTask
        ):
            self.opt = self.config.optimizer_gen(self.model, task)(
                self.model.parameters()
            )

            self.criterion = self.config.criterion(task)
            self.device = self.config.device

            if self.config.device != "cpu":
                self.criterion = self.criterion.to(self.config.device)

            data, model_classes = self.data_module.get_data(
                X_train, y_train, X_test, task
            )
            self.learn = Learner(
                self.model,
                self.opt,
                self.criterion,
                data,
                model_classes,
            )
            self.learn.device = self.device
            self.learn.model.to(self.device)
            gc.collect()

            self.runner = ModelRunner(cb_funcs=self.cbfs)

            # some additional default callbacks
            self.plot_loss = self.runner.cbs[1].plot_loss
            self.plot_lr = self.runner.cbs[1].plot_lr
            self.model_classes = model_classes

            self.learn.model.train()
            self.runner.fit(epochs=self.config.epoch_count, learn=self.learn)
            self.learn.model.eval()

            self.lrs = self.runner.cbs[1].lrs

            print("Loss", self.runner.loss)
        else:
            raise Exception("OpenML Task type not supported")
        return data, model_classes

    def add_callbacks(self):
        """
        Adds the user-defined callbacks to the list of callbacks
        """
        if self.callbacks is not None and len(self.callbacks) > 0:
            for callback in self.callbacks:
                if callback not in self.cbfs:
                    self.cbfs.append(callback)
                else:
                    # replace the callback with the new one in the same position
                    self.cbfs[self.cbfs.index(callback)] = callback

    def pred_test(self, task, model_copy, test_loader, predict_func):
        probabilities = []
        for batch_idx, inputs in enumerate(test_loader):
            inputs = self.config.sanitize(inputs)
            # if torch.cuda.is_available():
            inputs = inputs.to(self.config.device)

            # Perform inference on the batch
            pred_y_batch = model_copy(inputs)
            pred_y_batch = predict_func(pred_y_batch, task)
            pred_y_batch = pred_y_batch.cpu().detach().numpy()

            probabilities.append(pred_y_batch)

            # Concatenate probabilities from all batches
        pred_y = np.concatenate(probabilities, axis=0)
        return pred_y

# Define a trainer
class BasicTrainer:
    """
    BasicTrainer class provides a simple training loop for PyTorch models.You pass in the model, loss function, optimizer, data loaders, and device. The fit method trains the model for the specified number of epochs.
    """
    def __init__(self, model: Any, loss_fn: Any, opt: Any, dataloader_train: torch.utils.data.DataLoader, dataloader_test: torch.utils.data.DataLoader, device: torch.device):
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.opt = opt(self.model.parameters())
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.losses = {'train': [], 'test': []}

    def train_step(self, x, y):
        self.model.train()
        self.opt.zero_grad()
        yhat = self.model(x)
        loss = self.loss_fn(yhat, y)
        loss.backward()
        self.opt.step()
        return loss.item()

    def test_step(self, x, y):
        self.model.eval()
        with torch.no_grad():
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
        return loss.item()
    
    def fit(self, epochs):
        if self.dataloader_train is None:
            raise ValueError('dataloader_train is not set')
        if self.dataloader_test is None:
            raise ValueError('dataloader_test is not set')
        bar = tqdm(range(epochs), desc='Epochs')
        for epoch in bar:
            # train
            for x, y in self.dataloader_train:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.train_step(x, y)
                self.losses['train'].append(loss)
            # test
            test_loss = 0
            for x, y in self.dataloader_test:
                x, y = x.to(self.device), y.to(self.device)
                test_loss += self.test_step(x, y)
                self.losses['test'].append(test_loss)
            bar.set_postfix({'Train loss': loss, 'Test loss': test_loss, 'Epoch': epoch + 1})