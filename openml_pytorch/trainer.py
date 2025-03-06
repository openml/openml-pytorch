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
from typing import Any, Callable, List, Optional, Union

import netron
import numpy as np
import onnx
import openml
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

class DataContainer:
    """Container for training, validation, and test data loaders."""
    
    def __init__(self, train_dl: DataLoader, valid_dl: Optional[DataLoader], test_dl: DataLoader, model_classes: List[Any]):
        self.train_dl, self.valid_dl, self.test_dl = train_dl, valid_dl, test_dl
        self.model_classes = model_classes

    @property
    def train_ds(self): return self.train_dl.dataset
    
    @property
    def valid_ds(self): 
        if self.valid_dl is not None:
            return self.valid_dl.dataset
        else:
            return None
    
    @property
    def test_ds(self): return self.test_dl.dataset if self.test_dl else None


class DataModule:
    def __init__(
        self, batch_size: int, num_workers: int, target_mode: str, transforms=None
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_mode = target_mode
        self.transforms = transforms
        self.model_classes = None

    def _split_data(self, X, y, create_validation: bool):
        """
        Split the data into training, validation, and test sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_valid, y_valid = (None, None)

        if create_validation:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

        return X_train, y_train, X_test, y_test, X_valid, y_valid

    def _encode_labels(self, y_train, y_test, y_valid):
        """
        Encode the labels of the datasets using a label encoder if the target_mode is categorical.
        """
        if self.target_mode == "categorical":
            self.model_classes = (
                y_train.unique()
                if isinstance(y_train, pd.Series)
                else list(set(y_train))
            )
            return self.encode_labels_for_datasets(y_train, y_test, y_valid)
        return y_train, y_test, y_valid
    
    def _create_data_container(
        self,
        dataset_class,
        X_train,
        y_train,
        X_test,
        y_test,
        X_valid,
        y_valid,
        create_validation,
        model_classes,
        *args,
        **kwargs,
    ):
        """
        Create a DataContainer object from the provided datasets. This makes it easier to access the datasets in one object.
        """
        train_ds = dataset_class(X_train, y_train, *args,**kwargs, transform_x=self.transforms)
        test_ds = dataset_class(X_test, y_test, *args,**kwargs, transform_x=self.transforms)
        valid_ds = (
            dataset_class(X_valid, y_valid, *args,**kwargs,  transform_x=self.transforms)
            if create_validation
            else None
        )

        return DataContainer(
            train_dl=DataLoader(
                train_ds, batch_size=self.batch_size, num_workers=self.num_workers
            ),
            valid_dl=(
                DataLoader(
                    valid_ds, batch_size=self.batch_size, num_workers=self.num_workers
                )
                if valid_ds
                else None
            ),
            test_dl=DataLoader(
                test_ds, batch_size=self.batch_size, num_workers=self.num_workers
            ), model_classes = model_classes
        )

    def encode_labels_for_datasets(self, y_train, y_test, y_valid):
        """
        Encode the labels of the datasets using a label encoder if the target_mode is categorical.
        """
        label_encoder = preprocessing.LabelEncoder().fit(y_train)
        self.model_classes = label_encoder.classes_
        y_train = self.encode_labels(y_train, label_encoder=label_encoder)  # type: ignore
        if y_test is not None:
            y_test = self.encode_labels(y_test, label_encoder=label_encoder)
        if y_valid is not None:
            y_valid = self.encode_labels(y_valid, label_encoder=label_encoder)
        
        # convert to tensor
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        y_valid = torch.tensor(y_valid)

        return y_train, y_test, y_valid

    def encode_labels(
        self,
        y: Union[pd.Series, list, np.ndarray],
        label_encoder: preprocessing.LabelEncoder,
    ):
        """Encode the labels using the provided label encoder."""
        return label_encoder.transform(y)
    
    def load_tabular_openml_dataset(
        self, dataset_id: int, create_validation_dataset: bool = True
    ):
        """
        Load a tabular dataset from OpenML into a DataContainer object.
        """
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        if y is None:
            raise ValueError("Target variable 'y' cannot be None.")

        X_train, y_train, X_test, y_test, X_valid, y_valid = self._split_data(
            X, y, create_validation_dataset
        )
        y_train, y_test, y_valid = self._encode_labels(y_train, y_test, y_valid)

        self.model_classes = y_train.unique() if isinstance(y_train, pd.Series) else list(set(y_train))

        return self._create_data_container(
            OpenMLTabularDataset,
            X_train = X_train,
            y_train = y_train,
            X_test = X_test,
            y_test = y_test,
            X_valid = X_valid,
            y_valid = y_valid,
            create_validation = create_validation_dataset,
            model_classes=self.model_classes,
        )

    def load_image_openml_dataset(
        self,
        dataset_id: int,
        image_dir: str,
        image_size: int,
        filename_col: str,
        target_column: str,
        create_validation_dataset: bool = True,
    ):
        """
        Load an image dataset from OpenML into a DataContainer object."""
        dataset = openml.datasets.get_dataset(
            dataset_id, download_data=True
        ).get_data()[0]
        X, y = dataset[filename_col], dataset[target_column]
        X_train, y_train, X_test, y_test, X_valid, y_valid = self._split_data(
            X, y, create_validation_dataset
        )
        y_train, y_test, y_valid = self._encode_labels(y_train, y_test, y_valid)

        self.model_classes = y_train.unique() if isinstance(y_train, pd.Series) else list(set(y_train))

        return self._create_data_container(
            OpenMLImageDataset,
            X_train = X_train,
            y_train = y_train,
            X_test = X_test,
            y_test = y_test,
            X_valid = X_valid,
            y_valid = y_valid,
            create_validation = create_validation_dataset,
            image_dir = image_dir,
            image_size = image_size,
            model_classes=self.model_classes,
        )

    def load_tabular_openml_dataset_from_task_id(
        self, task_id: int, create_validation_dataset: bool = True
    ):
        """
        Load a tabular dataset from an OpenML task ID into a DataContainer object.
        """
        dataset_id = openml.tasks.get_task(task_id).dataset_id
        self.load_tabular_openml_dataset(dataset_id, create_validation_dataset)
    
    def load_image_openml_dataset_from_task_id(
        self,
        task_id: int,
        image_dir: str,
        image_size: int,
        filename_col: str,
        target_column: str,
        create_validation_dataset: bool = True,
    ):
        """
        Load an image dataset from an OpenML task ID into a DataContainer object.
        """
        dataset_id = openml.tasks.get_task(task_id).dataset_id
        self.load_image_openml_dataset(
            dataset_id,
            image_dir,
            image_size,
            filename_col,
            target_column,
            create_validation_dataset
        )
    
    def load_custom_dataloaders(self, train_dl: DataLoader, test_dl: DataLoader, valid_dl: Optional[DataLoader] = None):
        """
        Load the dataloaders from the provided data loaders into a DataContainer object.
        """
        y_train = train_dl.dataset.y
        self.model_classes = y_train.unique() if isinstance(y_train, pd.Series) else list(set(y_train))
        return DataContainer(train_dl, valid_dl, test_dl, self.model_classes)



# class Trainer:
#     def __init__(
#         self,
#         experiment_name: str,
#         dl,  # Removed DataContainer typing to avoid dependency issues
#         model: Any,
#         opt=torch.optim.Adam,
#         loss_fn=torch.nn.CrossEntropyLoss,
#         metrics: List[Callable] = [accuracy],
#         callbacks: List[Callback] = [],
#         use_tensorboard: bool = True,
#         device: torch.device = torch.device("cpu"),
#         **kwargs,
#     ):
#         self.experiment_name = experiment_name
#         self.data = dl  # Store dataset
#         self.model = model.to(device)
#         self.opt = opt(self.model.parameters())
#         self.loss_fn = loss_fn
#         self.metrics = metrics
#         self.device = device if device else self.get_device()

#         self.loss = 0
#         self.in_train = False
#         self.training_state = True

#         # Logging setup
#         self.logger: logging.Logger = logging.getLogger(__name__)
#         self.phases = [0.2, 0.8]
#         self.scheds = combine_scheds(self.phases, [sched_cos(1e-4, 1e-2), sched_cos(1e-3, 1e-5)])

#         # TensorBoard setup
#         self.use_tensorboard = use_tensorboard
#         self.tensorboard_writer = None
#         if use_tensorboard:
#             timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             self.tensorboard_writer = SummaryWriter(
#                 comment=experiment_name,
#                 log_dir=f"tensorboard_logs/{experiment_name}/{timestamp}",
#             )

#         # Callbacks setup
#         self.callbacks = callbacks
#         self.cbfs = [
#             Recorder,
#             partial(AvgStatsCallback, self.metrics),
#             partial(ParamScheduler, "lr", self.scheds),
#             partial(PutDataOnDeviceCallback, self.device),
#         ]
#         if self.tensorboard_writer:
#             self.cbfs.append(partial(TensorBoardCallback, self.tensorboard_writer))

#         self.add_callbacks()
#         self.cbs = [TrainEvalCallback()] + [cbf() for cbf in self.cbfs]

#     def get_device(self):
#         """Detects and returns the best available device (CUDA, MPS, or CPU)."""
#         if torch.cuda.is_available():
#             return torch.device("cuda")
#         elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
#             return torch.device("mps")
#         return torch.device("cpu")

#     def add_callbacks(self):
#         """Adds the user-defined callbacks to the list of callbacks."""
#         for callback in self.callbacks:
#             if callback not in self.cbfs:
#                 self.cbfs.append(callback)

#     def one_batch(self, xb, yb):
#         """Processes a single batch in training."""
#         try:
#             self.xb, self.yb = xb.to(self.device), yb.to(self.device)
#             self.model.train()
            
#             self("begin_batch")
#             global sample_input
#             sample_input = xb

#             self.pred = self.model(xb)
#             self("after_pred")

#             self.loss = self.loss_fn(self.pred, yb)
#             self("after_loss")

#             if self.in_train:
#                 self.loss.backward()
#                 self("after_backward")
#                 self.opt.step()
#                 self("after_step")
#                 self.opt.zero_grad()
#         except CancelBatchException:
#             self("after_cancel_batch")
#         finally:
#             self("after_batch")

#     def all_batches(self, dl):
#         """Processes all batches in an epoch."""
#         self.iters = len(dl)
#         try:
#             for xb, yb in tqdm(dl, leave=False):
#                 self.one_batch(xb, yb)
#         except CancelEpochException:
#             self("after_cancel_epoch")

#     def fit(self, epochs):
#         """Runs the full training process."""
#         self.epochs, self.loss = epochs, torch.tensor(0.0)
#         try:
#             for cb in self.cbs:
#                 cb.set_runner(self)
#             self("begin_fit")

#             for epoch in range(epochs):
#                 self.epoch = epoch
#                 self.in_train = True
#                 if not self("begin_epoch"):
#                     self.all_batches(self.data.train_dl)

#                 self.in_train = False
#                 with torch.no_grad():
#                     if not self("begin_validate"):
#                         self.all_batches(self.data.valid_dl)

#                 self("after_epoch")
#         except CancelTrainException:
#             self("after_cancel_train")
#         finally:
#             self("after_fit")
#             self.in_train = False

#     def export_to_onnx(self):
#         """Converts the model to ONNX format."""
#         global sample_input
#         f = io.BytesIO()
#         torch.onnx.export(self.model, sample_input, f)
#         return f.getvalue()

#     def export_to_netron(self, onnx_file_name="model.onnx"):
#         """Exports and serves the model using Netron."""
#         onnx_data = self.export_to_onnx()
#         with open(onnx_file_name, "wb") as f:
#             f.write(onnx_data)
#         print(f"ONNX model saved to {onnx_file_name}. Delete if needed.")
#         netron.start(onnx_file_name)

#     def _prediction_to_probabilities(self, y: np.ndarray, classes: List[Any]) -> np.ndarray:
#         """Transforms predicted probabilities to match OpenML class indices."""
#         if not isinstance(classes, list):
#             raise ValueError("Classes must be a list.")
#         result = np.zeros((len(y), len(classes)), dtype=np.float32)
#         for obs, prediction_idx in enumerate(y):
#             result[obs][prediction_idx] = 1.0
#         return result

#     def train(self, num_epochs: int):
#         """Runs the training loop."""
#         self.model.train()
#         self.fit(num_epochs)
#         self.model.eval()

#     def __call__(self, cb_name):
#         """Calls the appropriate callback function."""
#         res = False
#         for cb in sorted(self.cbs, key=lambda x: x._order):
#             res = cb(cb_name) and res
#         return res

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
    def loss_fn(self):
        return self.learn.loss_fn

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
            self.xb = self.xb.to(self.learn.device)
            self.yb = self.yb.to(self.learn.device)
            # Below two lines are hack to convert model to onnx
            global sample_input
            sample_input = self.xb
            self("begin_batch")
            self.pred = self.model(self.xb)
            self("after_pred")
            self.loss = self.loss_fn(self.pred, self.yb)
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
            for xb, yb in tqdm(dl, leave=False, desc="Batches"):
                self.one_batch(xb, yb)
        except CancelEpochException:
            self("after_cancel_epoch")

    def fit(self, epochs, learn):
        self.epochs, self.learn, self.loss = epochs, learn, torch.tensor(0.0)
        try:
            for cb in self.cbs:
                cb.set_runner(self)
            self("begin_fit")
            for epoch in tqdm(range(epochs), leave=False, desc= "Epochs"):
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
    def __init__(self, model, opt, loss_fn, data, model_classes):
        (
            self.model,
            self.opt,
            self.loss_fn,
            self.data,
            self.model_classes,
        ) = (model, opt, loss_fn, data, model_classes)

class Trainer:
    def __init__(self, experiment_name:str, task_type:str, dl:DataContainer, model: Any, opt, loss_fn, metrics: List[Callable], callbacks: List[Callback], use_tensorboard: bool = True, device: torch.device = torch.device("cpu")):
        self.experiment_name = experiment_name
        self.dl = dl
        self.model = model
        self.opt = opt
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.device = device if device else self.get_device()
        self.loss = 0
        self.in_train = False
        self.training_state = True
        self.task_type = task_type
        self.logger = logging.getLogger(__name__)
        self.use_tensorboard = use_tensorboard
        self.callbacks = callbacks
        self.onnx_model = None

        # Tensorboard support
        self.tensorboard_writer = None
        if use_tensorboard:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.tensorboard_writer = SummaryWriter(
                comment=experiment_name,
                log_dir=f"tensorboard_logs/{experiment_name}/{timestamp}",
            )
        
        self.phases = [0.2, 0.8]
        self.scheds = combine_scheds(
            self.phases, [sched_cos(1e-4, 1e-2), sched_cos(1e-3, 1e-5)]
        )

        # Add default callbacks
        self.cbfs = [
            Recorder,
            partial(AvgStatsCallback, self.metrics),
            partial(ParamScheduler, "lr", self.scheds),
            partial(PutDataOnDeviceCallback, self.device),
        ]
        if self.tensorboard_writer is not None:
            self.cbfs.append(partial(TensorBoardCallback, self.tensorboard_writer))

        self.add_callbacks()
    
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

        # write the onnx model to a file
        with open(onnx_file_name, "wb") as f:
            f.write(self.onnx_model)
            print(f"Writing onnx model to {onnx_file_name}. Delete if neeeded")

        # serve with netro
        netron.start(onnx_file_name)
    
    def train(self, epoch_count):
        """
        Runs the training loop.
        """
        self.opt = self.opt(self.model.parameters())

        self.learn = Learner(
            model=self.model,
            opt=self.opt,
            loss_fn=self.loss_fn,
            data=self.dl,
            model_classes=self.dl.model_classes,
        )
        self.learn.device = self.device

        self.runner = ModelRunner(cb_funcs=self.cbfs)

        # some additional default callbacks
        self.plot_loss = self.runner.cbs[1].plot_loss
        self.plot_lr = self.runner.cbs[1].plot_lr
        # self.lrs = self.runner.cbs[1].lrs
        # self.losses = self.runner.cbs[1].losses

        self.learn.model.train()
        self.runner.fit(epochs=epoch_count, learn=self.learn)
        self.learn.model.eval()

        self.evaluate()
    
    def evaluate(self):
        """
        Evaluates the model on the test set.
        """
        self.model_copy = copy.deepcopy(self.model)
        if self.task_type == "regression":
            self.model.eval()
            pred_y = self.pred_test(self.task_type, self.model_copy, self.dl.test_dl, self._default_predict)
        
        elif self.task_type == "classification":
            self.model.eval()
            self.pred_y = self.pred_test(self.task_type, self.model_copy, self.dl.test_dl, self._default_predict)
            self.pred_y_proba = self.pred_test(self.task_type, self.model_copy, self.dl.test_dl, self._default_predict_proba)

        return self.pred_y, self.pred_y_proba
    
    def _default_predict(self,output: torch.Tensor, task: str) -> torch.Tensor:
        """
        _default_predict turns the outputs into predictions by returning the argmax of the output tensor for classification tasks, and by flattening the prediction in case of the regression
        """
        output_axis = output.dim() - 1
        if task == "classification":
            output = torch.argmax(output, dim=output_axis)
        elif task == "regression":
            output = output.view(-1)
        else:
            raise ValueError(task)
        return output

    def _default_predict_proba(self,
        output: torch.Tensor, task: str
    ) -> torch.Tensor:
        """
        _default_predict_proba turns the outputs into probabilities using softmax
        """
        output_axis = output.dim() - 1
        output = output.softmax(dim=output_axis)
        return output
    
    def sanitize(self,tensor):
        return torch.where(
            torch.isnan(tensor), torch.ones_like(tensor) * torch.tensor(1e-6), tensor)
    
    def pred_test(self, task, model_copy, test_loader, predict_func):
        probabilities = []
        for batch_idx, inputs in enumerate(test_loader):
            inputs = inputs[0]
            
            inputs = self.sanitize(inputs)
            # if torch.cuda.is_available():
            inputs = inputs.to(self.device)

            # perform inference
            with torch.no_grad():
                output = model_copy(inputs)
                probabilities.append(predict_func(output, task))
        return torch.cat(probabilities).cpu().numpy()
    
#   def _run_model_on_fold(
#         self,
#         model: Any,
#         task: "OpenMLTask",
#         X_train: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame],
#         rep_no: int,
#         fold_no: int,
#         y_train: Optional[np.ndarray] = None,
#         X_test: Optional[Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]] = None,
#     ) -> Tuple[
#         np.ndarray,
#         np.ndarray,
#         "OrderedDict[str, float]",
#         Optional[OpenMLRunTrace],
#         Optional[Any],
#     ]: 
#         

    def run_model_on_fold(self, model: Any, task: OpenMLTask, X_train: Union[np.ndarray, pd.DataFrame], rep_no: int, fold_no: int, y_train: Optional[np.ndarray] = None, X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None):
        """
        Runs the model on a single fold of the data.

        Parameters
        ----------
        model : Any
            The model to be trained and evaluated.
        task : OpenMLTask
            The task to be performed.
        X_train : Union[np.ndarray, pd.DataFrame]
            The training data.
        rep_no : int
            The repetition number.
        fold_no : int
            The fold number.
        y_train : Optional[np.ndarray], optional
            The training labels, by default None.
        X_test : Optional[Union[np.ndarray, pd.DataFrame]], optional
            The test data, by default None.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, OrderedDict[str, float], Optional[OpenMLRunTrace], Optional
        """
        # set the model to evaluation
        model.eval()
        # get the data
        if y_train is not None:
            y_train = torch.tensor(y_train)
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        # get the data loader
        train_dl = DataLoader(X_train, batch_size=32, shuffle=True)
        test_dl = DataLoader(X_test, batch_size=32, shuffle=False)
        # create the learner

        learn = Learner(
            model=model,
            opt=torch.optim.Adam(model.parameters()),
            loss_fn=torch.nn.CrossEntropyLoss(),
            data=DataContainer(train_dl, None, test_dl, None),
            model_classes=None
        )
        # create the runner
        runner = ModelRunner()
        # fit the model
        runner.fit(1, learn)
        # evaluate the model
        pred_y, pred_y_proba = runner.evaluate()
        return pred_y, pred_y_proba