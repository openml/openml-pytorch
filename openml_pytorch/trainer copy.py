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
    
    def __init__(self, train_dl: DataLoader, valid_dl: Optional[DataLoader], test_dl: DataLoader):
        self.train_dl, self.valid_dl, self.test_dl = train_dl, valid_dl, test_dl

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
        *args
    ):
        """
        Create a DataContainer object from the provided datasets. This makes it easier to access the datasets in one object.
        """
        train_ds = dataset_class(X_train, y_train, *args, transform_x=self.transforms)
        test_ds = dataset_class(X_test, y_test, *args, transform_x=self.transforms)
        valid_ds = (
            dataset_class(X_valid, y_valid, *args, transform_x=self.transforms)
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
            ),
        )

    def encode_labels_for_datasets(self, y_train, y_test, y_valid):
        """
        Encode the labels of the datasets using a label encoder if the target_mode is categorical.
        """
        label_encoder = preprocessing.LabelEncoder().fit(y_train)
        y_train = self.encode_labels(y_train, label_encoder=label_encoder)  # type: ignore
        if y_test is not None:
            y_test = self.encode_labels(y_test, label_encoder=label_encoder)
        if y_valid is not None:
            y_valid = self.encode_labels(y_valid, label_encoder=label_encoder)

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

        return self._create_data_container(
            OpenMLTabularDataset,
            X_train,
            y_train,
            X_test,
            y_test,
            X_valid,
            y_valid,
            create_validation_dataset,
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

        return self._create_data_container(
            OpenMLImageDataset,
            X_train,
            y_train,
            X_test,
            y_test,
            X_valid,
            y_valid,
            create_validation_dataset,
            image_dir,
            image_size,
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
        return DataContainer(train_dl, valid_dl, test_dl)


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
        model: Any,
        callbacks: List[Callback] = [],
        use_tensorboard: bool = True,
        **kwargs,
    ):
        self.experiment_name = experiment_name
        self.data_module = data_module
        self.model = model

        self.data_config = self.data_module.config
        self.task_type = self.data_config.task_type

        self.model_config = ModelConfigGenerator(self.task_type).return_model_config()

        self.callbacks = callbacks

        self.config = SimpleNamespace(
            **{**self.model_config.__dict__, **self.data_module.config.__dict__}
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
            # data, model_classes = self.run_training(task, X_train, y_train, X_test)
            data, model_classes = self.data_module.data_container, self.data_module.model_classes
            if isinstance(task, OpenMLSupervisedTask) or isinstance(
            task, OpenMLClassificationTask):
                self.run_training()
            else:
                raise Exception("OpenML Task type not supported") 

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

    def run_training(self, epoch_count):
        
        self.opt = self.config.optimizer_gen(self.model, task)(
            self.model.parameters()
        )

        self.criterion = self.config.criterion(task)
        self.device = self.config.device

        if self.config.device != "cpu":
            self.criterion = self.criterion.to(self.config.device)

        self.learn = Learner(
            self.model,
            self.opt,
            self.criterion,
            self.data_module.data_container,
            self.data_module.model_classes
        )
        self.learn.device = self.device
        self.learn.model.to(self.device)
        gc.collect()

        self.runner = ModelRunner(cb_funcs=self.cbfs)

        # some additional default callbacks
        self.plot_loss = self.runner.cbs[0].plot_loss
        self.plot_lr = self.runner.cbs[0].plot_lr

        self.learn.model.train()
        self.runner.fit(epochs=epoch_count, learn=self.learn)
        self.learn.model.eval()

        self.lrs = self.runner.cbs[1].lrs

        print("Loss", self.runner.loss)

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
