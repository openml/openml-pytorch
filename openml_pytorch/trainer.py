import logging
from types import SimpleNamespace
import warnings
import numpy as np
import torch
from typing import Any, List, Optional, Tuple, Union
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from openml.tasks import (
    OpenMLTask,
    OpenMLSupervisedTask,
    OpenMLClassificationTask,
    OpenMLRegressionTask,
)
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Lambda
import pandas as pd
import copy
import io
import onnx
from tqdm import tqdm
from .data import OpenMLImageDataset
from openml.exceptions import PyOpenMLError
from types import SimpleNamespace


class DefaultConfigGenerator:
    def _default_criterion_gen(self, task: OpenMLTask) -> torch.nn.Module:
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

    def _default_optimizer_gen(self, model: torch.nn.Module, _: OpenMLTask):
        """
        _default_optimizer_gen returns the torch.optim.Adam optimizer for the given model
        """
        return torch.optim.Adam

    def _default_scheduler_gen(self, optim, _: OpenMLTask) -> Any:
        """
        _default_scheduler_gen returns the torch.optim.lr_scheduler.ReduceLROnPlateau scheduler for the given optimizer
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim)

    def _default_predict(self, output: torch.Tensor, task: OpenMLTask) -> torch.Tensor:
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

    def _default_predict_proba(
        self, output: torch.Tensor, task: OpenMLTask
    ) -> torch.Tensor:
        """
        _default_predict_proba turns the outputs into probabilities using softmax
        """
        output_axis = output.dim() - 1
        output = output.softmax(dim=output_axis)
        return output

    def _default_sanitize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        _default sanitizer replaces NaNs with 1e-6
        """
        tensor = torch.where(
            torch.isnan(tensor), torch.ones_like(tensor) * torch.tensor(1e-6), tensor
        )
        return tensor

    def _default_retype_labels(
        self, tensor: torch.Tensor, task: OpenMLTask
    ) -> torch.Tensor:
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

    def return_model_config(self):
        """
        Returns a configuration object for the model
        de

        """

        return SimpleNamespace(
            device=self.get_device(),
            criterion_gen=self._default_criterion_gen,
            optimizer_gen=self._default_optimizer_gen,
            scheduler_gen=self._default_scheduler_gen,
            perform_validation=False,
            # predict turns the outputs of the model into actual predictions
            predict=self._default_predict,  # type: Callable[[torch.Tensor, OpenMLTask], torch.Tensor]
            # predict_proba turns the outputs of the model into probabilities for each class
            predict_proba=self._default_predict_proba,  # type: Callable[[torch.Tensor], torch.Tensor]
            # epoch_count represents the number of epochs the model should be trained for
            epoch_count=3,  # type: int,
            validation_split=0.1,
            # progress_callback=(
            #     self._default_progress_callback
            # ),  # type: Callable[[int, int, int, int, float, float], None]
            # enable progress bar
            verbose=True,
        )

    def return_data_config(self):
        return SimpleNamespace(
            type_of_data="image",
            # progress_callback is called when a training step is finished, in order to report the current progress
            # sanitize sanitizes the input data in order to ensure that models can be trained safely
            sanitize=self._default_sanitize,  # type: Callable[[torch.Tensor], torch.Tensor]
            # retype_labels changes the types of the labels in order to ensure type compatibility
            retype_labels=(
                self._default_retype_labels
            ),  # type: Callable[[torch.Tensor, OpenMLTask], torch.Tensor]
            # TODO: refactor all these later to the dataloader
            # image_size is the size of the images that are fed into the model
            image_size=128,
            # batch_size represents the processing batch size for training
            batch_size=64,  # type: int
            data_augmentation=None,
        )


class OpenMLDataModule:
    def __init__(
        self,
        type_of_data="image",
        filename_col="Filename",
        file_dir="images",
        target_mode="categorical",
        **kwargs,
    ):
        self.type_of_data = type_of_data
        self.config_gen = DefaultConfigGenerator()
        self.data_config = self.config_gen.return_data_config()

        self.data_config.filename_col = filename_col
        self.data_config.file_dir = file_dir
        self.data_config.target_mode = target_mode


class Callback:
    def __init__(self):
        self.data = None

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_loss_begin(self):
        pass

    def on_loss_end(self):
        pass

    def on_opt_step_begin(self):
        pass

    def on_opt_step_end(self):
        pass

    def on_scheduler_step_begin(self):
        pass

    def on_scheduler_step_end(self):
        pass


class LoggingCallback(Callback):
    def __init__(self, logger: logging.Logger, print_output: bool = False):
        super().__init__()
        self.logger = logger
        self.print_output = print_output

    def log_or_print(self, message: str):
        if self.print_output:
            print(message)
        self.logger.info(message)

    def on_train_begin(self):
        self.log_or_print("Training started")

    def on_train_end(self):
        self.log_or_print("Training ended")

    def on_epoch_begin(self):
        self.log_or_print("Epoch started")

    def on_epoch_end(self):
        self.log_or_print("Epoch ended")

    def on_batch_begin(self):
        self.log_or_print("Batch started")

    def on_batch_end(self):
        self.log_or_print("Batch ended")

    def on_loss_begin(self):
        self.log_or_print("Loss calculation started")

    def on_loss_end(self):
        self.log_or_print("Loss calculation ended")

    def on_opt_step_begin(self):
        self.log_or_print("Optimizer step started")

    def on_opt_step_end(self):
        self.log_or_print("Optimizer step ended")

    def on_scheduler_step_begin(self):
        self.log_or_print("Scheduler step started")

    def on_scheduler_step_end(self):
        self.log_or_print("Scheduler step ended")


class OpenMLTrainerModule:
    def __init__(
        self,
        data_module: OpenMLDataModule,
        callbacks: List[Callback] = [],
        **kwargs,
    ):
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
        self.callbacks.append(LoggingCallback(self.logger, print_output=False))
        self.loss = 0
        self.training_state = True

    def call_callbacks(self, method_name):
        for callback in self.callbacks:
            callback.data = self
            cb = getattr(callback, method_name)
            cb()

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

    def check_config(self):
        raise NotImplementedError

    def convert_to_rgb(self, image):
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def openml2pytorch_data(self, X, y, task) -> Any:
        # convert openml dataset to pytorch compatible dataset
        if self.data_module.type_of_data == "image":
            df = X
            columns_to_use = [self.config.filename_col]

            if y is not None:
                label_encoder = preprocessing.LabelEncoder().fit(y)
                df.loc[:, "encoded_labels"] = label_encoder.transform(y)
                label_mapping = {
                    index: label for index, label in enumerate(label_encoder.classes_)
                }
                columns_to_use = [self.config.filename_col, "encoded_labels"]
            else:
                label_mapping = None

            data = OpenMLImageDataset(
                image_size=self.config.image_size,
                annotations_df=df[columns_to_use],
                img_dir=self.config.file_dir,
                transform=Compose(
                    [
                        ToPILImage(),  # Convert tensor to PIL Image to ensure PIL Image operations can be applied.
                        Lambda(
                            self.convert_to_rgb
                        ),  # Convert PIL Image to RGB if it's not already.
                        Resize(
                            (self.config.image_size, self.config.image_size)
                        ),  # Resize the image.
                        ToTensor(),  # Convert the PIL Image back to a tensor.
                    ]
                ),
            )

            return data, label_mapping
        else:
            raise ValueError("Data type not supported")

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
                "please convert model classes to list prior to " "calling this fn"
            )
        result = np.zeros((len(y), len(classes)), dtype=np.float32)
        for obs, prediction_idx in enumerate(y):
            result[obs][prediction_idx] = 1.0
        return result

    # def run_model_on_fold(self, model, task, X_train, rep_no, fold_no, y_train, X_test):
    def run_model_on_fold(
        self,
        model: torch.nn.Module,
        task: OpenMLTask,
        X_train: pd.DataFrame,
        rep_no: int,
        fold_no: int,
        y_train: Optional[pd.Series],
        X_test: pd.DataFrame,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], OrderedDict, Optional[Any]]:

        self.model = copy.deepcopy(model).to(self.config.device)
        try:
            if isinstance(task, OpenMLSupervisedTask) or isinstance(
                task, OpenMLClassificationTask
            ):
  
                self.fold_no = fold_no
                self.rep_no = rep_no

                self.optimizer = self.config.optimizer_gen(self.model, task)(
                    self.model.parameters()
                )
                self.scheduler = self.config.scheduler_gen(self.optimizer, task)

                self.criterion = self.config.criterion_gen(task)
                self.pin_memory = False

                # if torch.cuda.is_available():
                if self.config.device != "cpu":
                    self.criterion = self.criterion.to(self.config.device)
                    self.pin_memory = True

                if self.config.perform_validation:
                    from sklearn.model_selection import train_test_split

                    # TODO: Here we're assuming that X has a label column, this won't work in general
                    X_train_train, x_val, y_train_train, y_val = train_test_split(
                        X_train,
                        y_train,
                        test_size=self.config.validation_split,
                        shuffle=True,
                        stratify=y_train,
                        random_state=0,
                    )
                    train, label_mapping = self.openml2pytorch_data(
                        X_train_train, y_train_train, task
                    )
                    train_loader = torch.utils.data.DataLoader(
                        train,
                        batch_size=self.config.batch_size,
                        shuffle=True,
                        pin_memory=self.pin_memory,
                    )

                    val, _ = self.openml2pytorch_data(x_val, None, task)
                    val_loader = torch.utils.data.DataLoader(
                        val,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        pin_memory=self.pin_memory,
                    )

                else:

                    train, label_mapping = self.openml2pytorch_data(
                        X_train, y_train, task
                    )
                    train_loader = torch.utils.data.DataLoader(
                        train,
                        batch_size=self.config.batch_size,
                        shuffle=True,
                        pin_memory=self.pin_memory,
                    )

                # we can disable tqdm but not enable it because that is how the API works. self.config.verbose is True by default. (So we need the opposite of the user input)
                disable_progress_bar = not self.config.verbose

                self.call_callbacks("on_train_begin")
                for epoch in tqdm(
                    range(self.config.epoch_count),
                    disable=disable_progress_bar,
                    desc="Epochs",
                ):
                    if self.training_state == True:
                        self.epoch = epoch
                        self.call_callbacks("on_epoch_begin")
                        correct = 0
                        incorrect = 0
                        running_loss = 0.0

                        self.model.train()

                        for batch_idx, (inputs, labels) in enumerate(train_loader):
                            self.call_callbacks("on_batch_begin")
                            inputs = self.config.sanitize(inputs)

                            # if torch.cuda.is_available():
                            inputs = inputs.to(self.config.device)
                            labels = labels.to(self.config.device)

                            # Below two lines are hack to convert model to onnx
                            global sample_input
                            sample_input = inputs

                            def _optimizer_step():
                                outputs = self.model(inputs)
                                self.call_callbacks("on_loss_begin")
                                self.loss = self.criterion(outputs, labels)
                                self.loss.backward()
                                self.call_callbacks("on_loss_end")
                                self.optimizer.zero_grad()
                                return self.loss

                            if labels.dtype != torch.int64:
                                labels = torch.tensor(
                                    labels, dtype=torch.long, device=labels.device
                                )

                            self.call_callbacks("on_opt_step_begin")
                            self.loss_opt = self.optimizer.step(_optimizer_step)
                            self.call_callbacks("on_opt_step_end")

                            self.scheduler.step(self.loss_opt)

                            predicted = self.model(inputs)
                            predicted = self.config.predict(predicted, task)

                            accuracy = float("nan")  # type: float
                            if isinstance(task, OpenMLClassificationTask):
                                correct += (predicted == labels).sum()
                                incorrect += (predicted != labels).sum()
                                accuracy_tensor = (
                                    torch.tensor(1.0) * correct / (correct + incorrect)
                                )
                                accuracy = accuracy_tensor.item()

                            # Print training progress information
                            running_loss += self.loss_opt.item()
                            if batch_idx % 100 == 99:  #  print every 100 mini-batches
                                print(
                                    f"Epoch: {epoch + 1}, Batch: {batch_idx + 1:5d}, Loss: {running_loss / 100:.3f}"
                                )
                                running_loss = 0.0

                            self.config.progress_callback(
                                fold_no,
                                rep_no,
                                epoch,
                                batch_idx,
                                self.loss_opt.item(),
                                accuracy,
                            )
                            self.call_callbacks("on_batch_end")

                    # validation phase
                    if self.config.perform_validation:

                        self.model.eval()
                        correct_val = 0
                        incorrect_val = 0
                        val_loss = 0

                        with torch.no_grad():
                            for inputs_val, labels_val in enumerate(val_loader):

                                # if torch.cuda.is_available():
                                inputs_val = inputs.to(self.config.device)
                                labels_val = labels.to(self.config.device)
                                outputs_val = self.model(inputs_val)
                                if labels_val.dtype != torch.int64:
                                    labels_val = torch.tensor(
                                        labels_val,
                                        dtype=torch.long,
                                        device=labels.device,
                                    )
                                loss_val = self.criterion(outputs_val, labels_val)

                                predicted_val = self.config.predict(outputs_val, task)
                                correct_val += (
                                    (predicted_val == labels_val).sum().item()
                                )
                                incorrect_val += (
                                    (predicted_val != labels_val).sum().item()
                                )

                                val_loss += loss_val.item()

                        accuracy_val = correct_val / (correct_val + incorrect_val)

                        # Print validation metrics
                        print(
                            f"Epoch: {epoch + 1}, Validation Loss: {val_loss / len(val_loader):.3f}, Validation Accuracy: {accuracy_val:.3f}"
                        )
                    self.call_callbacks("on_epoch_end")
                self.call_callbacks("on_train_end")

        except AttributeError as e:
            # typically happens when training a regressor8 on classification task
            raise PyOpenMLError(str(e))

        if isinstance(task, OpenMLClassificationTask):
            # Convert class labels to numerical indices

            x_train_labels = (
                X_train_train["encoded_labels"]
                if self.config.perform_validation
                else (
                    X_train["Class_encoded"]
                    if "Class_encoded" in X_train
                    else X_train["encoded_labels"]
                )
            )
            model_classes = np.sort(x_train_labels.astype("int").unique())
            # model_classes = np.amax(y_train)

        # In supervised learning this returns the predictions for Y
        if isinstance(task, OpenMLSupervisedTask):
            self.model.eval()

            # name = task.get_dataset().name
            # dataset_name = name.split('Meta_Album_')[1] if 'Meta_Album' in name else name

            test, _ = self.openml2pytorch_data(X_test, None, task)
            test_loader = torch.utils.data.DataLoader(
                test,
                batch_size=self.config.batch_size,
                shuffle=False,
                pin_memory=self.config.device != "cpu",
            )
            pred_y = self.pred_test(task, self.model, test_loader, self.config.predict)
        else:
            raise ValueError(task)

        if isinstance(task, OpenMLClassificationTask):

            try:
                self.model.eval()

                test, _ = self.openml2pytorch_data(X_test, None, task)
                test_loader = torch.utils.data.DataLoader(
                    test,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    pin_memory=self.config.device != "cpu",
                )

                proba_y = self.pred_test(
                    task, self.model, test_loader, self.config.predict_proba
                )

            except AttributeError:
                if task.class_labels is not None:
                    proba_y = self._prediction_to_probabilities(
                        pred_y, list(task.class_labels)
                    )
                else:
                    raise ValueError("The task has no class labels")

            if task.class_labels is None:
                task.class_labels = list(label_mapping.values())

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

        # Convert model to onnx
        onnx_ = self._onnx_export(self.model)

        global last_models
        last_models = onnx_

        return pred_y, proba_y, self.user_defined_measures, None

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

    def _onnx_export(self, model_copy):
        f = io.BytesIO()
        torch.onnx.export(model_copy, sample_input, f)
        onnx_model = onnx.load_model_from_string(f.getvalue())
        onnx_ = onnx_model.SerializeToString()
        return onnx_
