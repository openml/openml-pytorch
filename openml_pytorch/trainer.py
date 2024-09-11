import gc
import logging
import re
from types import SimpleNamespace
import warnings
import numpy as np
import torch
from typing import Any, Iterable, List, Optional, Tuple, Union
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from openml.tasks import (
    OpenMLTask,
    OpenMLSupervisedTask,
    OpenMLClassificationTask,
    OpenMLRegressionTask,
)
import torch.amp
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Lambda
import pandas as pd
import copy
import io
import onnx
from tqdm import tqdm
from .data import OpenMLImageDataset
from openml.exceptions import PyOpenMLError
from types import SimpleNamespace
import matplotlib.pyplot as plt
from functools import partial
import math

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos): return start + pos*(end-start)

@annealer
def sched_cos(start, end, pos): return start + (1+math.cos(math.pi*(1-pos)))*(end-start)/2

@annealer
def sched_no(start, end, pos): return start

@annealer
def sched_exp(start, end, pos): return start*(end/start)**pos

torch.Tensor.ndim = property(lambda x: len(x.shape))

def combine_scheds(pcts, scheds):
    assert sum(pcts)==1.
    pcts = torch.tensor([0] + listify(pcts))
    assert torch.all(pcts>=0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos>=pcts).nonzero().max()
        actual_pos = (pos-pcts[idx])/(pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner

# def cos_1cycle_anneal(start, high, end):
#     return [sched_cos(start, high), sched_cos(high, end)]
def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb.long()).float().mean()

class Callback():
    _order = 0
    def set_runner(self, run): self.run = run
    def __getattr__(self, k): return getattr(self.run, k)
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False

class ParamScheduler(Callback):
    _order = 1
    def __init__(self, pname, sched_funcs): self.pname, self.sched_funcs = pname, sched_funcs
    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs]*len(self.opt.param_groups)
    def set_param(self):
        assert len(self.opt.param_groups)==len(self.sched_funcs)
        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = f(self.n_epochs/self.epochs)
    def begin_batch(self):
        if self.in_train: self.set_param()

class Recorder(Callback):
    def begin_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []
    def after_batch(self):
        if not self.in_train: return
        for pg, lr in zip(self.opt.param_groups, self.lrs): lr.append(pg['lr'])
        self.losses.append(self.loss.detach().cpu())
    
    def plot_lr(self, pgid=-1): plt.plot(self.lrs[pgid])
    def plot_loss(self, skip_last=0): plt.plot(self.losse[:len(self.losses)-skip_last])
    def plot(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[pgid]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(lrs[:n], losses[:n])

def listify(o):
    if o is None: return [] 
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

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



class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs = 0
        self.run.n_iter = 0
    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs+=1./self.iters
        self.run.n_iter+=1
    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True
    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass

class AvgStats():
    def __init__(self, metrics, in_train): self.metrics, self.in_train = listify(metrics), in_train
    def reset(self):
        self.tot_loss, self.count = 0., 0
        self.tot_mets = [0.]*len(self.metrics)
    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]
    
    def __repr__(self):
        if not self.count: return ''
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"
    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss+=run.loss*bn
        self.count+=bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i]+=m(run.pred, run.yb)*bn

class AvgStatsCallBack(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats = AvgStats(metrics, True), AvgStats(metrics, False)
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)

class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop, self.cbs = False, [TrainEvalCallback()] + cbs

    @property
    def opt(self): return self.learn.opt
    @property
    def model(self): return self.learn.model
    @property
    def criterion(self): return self.learn.criterion
    @property
    def data(self): return self.learn.data
    @property
    def label_mapping(self): return self.learn.label_mapping
    @property
    def model_classes(self): return self.learn.model_classes
    
    def one_batch(self, xb, yb):
        try: 
            self.xb, self.yb = xb, yb
            self.xb = self.xb.to(self.learn.device)
            self.yb = self.yb.to(self.learn.device)
            # Below two lines are hack to convert model to onnx
            global sample_input
            sample_input = self.xb
            self('begin_batch')
            self.pred = self.model(self.xb)
            self('after_pred')
            self.loss = self.criterion(self.pred, self.yb)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally: self('after_batch')
    
    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb, yb in tqdm(dl, leave=False): self.one_batch(xb, yb)
        except CancelEpochException: self('after_cancel_epoch')
    def fit(self, epochs, learn):
        self.epochs, self.learn, self.loss = epochs, learn, torch.tensor(0.)
        try: 
            for cb in self.cbs: cb.set_runner(self)
            self('begin_fit')
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.data.train_dl)
                with torch.no_grad():
                    if not self('begin_validate'): self.all_batches(self.data.valid_dl)
                self('after_epoch')
        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.learn = None
    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) and res
        return res
    
class Learner():
    def __init__(self, model, opt, criterion, data, label_mapping, model_classes):
        self.model, self.opt, self.criterion, self.data, self.label_mapping, self.model_classes = model, opt, criterion, data, label_mapping, model_classes

class DataBunch():
    def __init__(self, train_dl, valid_dl, test_dl = None):
        self.train_dl, self.valid_dl = train_dl, valid_dl
        self.test_dl = test_dl

    @property
    def train_ds(self): return self.train_dl.dataset
    
    @property
    def valid_ds(self): return self.valid_dl.dataset

    @property
    def test_ds(self): return self.test_dl.dataset

# def get_data(train_idx, valid_idx):
#     x_train_ds = torch.tensor(train_x[train_idx], dtype=torch.long).to(device)
#     y_train_ds = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).to(device)
#     x_val_ds = torch.tensor(train_x[valid_idx], dtype=torch.long).to(device)
#     y_val_ds = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).to(device)
#     train_ds = torch.utils.data.TensorDataset(x_train_ds, y_train_ds)
#     valid_ds = torch.utils.data.TensorDataset(x_val_ds, y_val_ds)
#     train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
#     data = DataBunch(train_dl, valid_dl)
#     return data

# loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
# def get_learner(train_idx, valid_idx):
#     data = get_data(train_idx, valid_idx)
# get model - model, opt
#     learn = Learner(*get_model(data), loss_fn, data=data)
#     return learn
# def train_and_eval():
#     test_preds = np.zeros(len(test_x))
#     for fold, (train_idx, valid_idx) in enumerate(splits):
#         print('Fold:', fold)
#         torch.cuda.empty_cache()
#         learn = get_learner(train_idx, valid_idx)
#         gc.collect()
#         run = Runner(cb_funcs=cbfs)
#         learn.model.train()
#         run.fit(4, learn)
#         learn.model.eval()
#         test_preds_fold = np.zeros(len(test_dl.dataset))
#         for i, (x_batch,) in enumerate(test_dl):
#             with torch.no_grad():
#                 y_pred = learn.model(x_batch).detach()
#             test_preds_fold[i*batch_size:(i+1)*batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
#         test_preds+=test_preds_fold/len(splits)
#         del(learn)
#         gc.collect()
#         print(f'Test {fold} added')
#     print('Training Completed')
#     return test_preds



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
        # self.callbacks.append(LoggingCallback(self.logger, print_output=False))
        self.loss = 0
        self.training_state = True

    def sigmoid(self,x): return 1/(1+np.exp(-x)) 

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
    
    def get_data(self,X_train:pd.DataFrame, y_train:Optional[pd.Series], X_test: pd.DataFrame, task):
        from sklearn.model_selection import train_test_split

        # TODO: Here we're assuming that X has a label column, this won't work in general

        # train/val loader
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
            # pin_memory=self.pin_memory,
        )

        val, _ = self.openml2pytorch_data(x_val, y_val, task)
        val_loader = torch.utils.data.DataLoader(
            val,
            batch_size=self.config.batch_size,
            shuffle=False,
            # pin_memory=self.pin_memory,
        )

        # test loader
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

        # In supervised learning this returns the predictions for Y
        if isinstance(task, OpenMLSupervisedTask):
            # name = task.get_dataset().name
            # dataset_name = name.split('Meta_Album_')[1] if 'Meta_Album' in name else name

            test, _ = self.openml2pytorch_data(X_test, None, task)
            test_loader = torch.utils.data.DataLoader(
                test,
                batch_size=self.config.batch_size,
                shuffle=False,
                # pin_memory=self.config.device != "cpu",
            )
        
        else:
            raise ValueError(task)
        
        return DataBunch(train_loader, val_loader, test_loader), label_mapping, model_classes
    

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
        # self.config.device = "cpu" #FIx this

        self.model = copy.deepcopy(model)
        phases = [0.2, 0.8]
        scheds = combine_scheds(phases, [sched_cos(1e-4, 5e-3), sched_cos(5e-3, 1e-3)])

        self.cbfs = [Recorder, partial(AvgStatsCallBack, accuracy), partial(ParamScheduler, 'lr', scheds)]
        try:
            if isinstance(task, OpenMLSupervisedTask) or isinstance(
                task, OpenMLClassificationTask
            ):
  
                self.fold_no = fold_no
                self.rep_no = rep_no

                self.opt = self.config.optimizer_gen(self.model, task)(
                    self.model.parameters()
                )
                # self.scheduler = self.config.scheduler_gen(self.opt, task)

                self.criterion = self.config.criterion_gen(task)
                self.pin_memory = False
                self.device = self.config.device

                # if torch.cuda.is_available():
                if self.config.device != "cpu":
                    self.criterion = self.criterion.to(self.config.device)
                    self.pin_memory = True

                # we can disable tqdm but not enable it because that is how the API works. self.config.verbose is True by default. (So we need the opposite of the user input)
                disable_progress_bar = not self.config.verbose

                # train_loader, val_loader, test_loader, label_mapping, model_classes = self.get_data(X_train, y_train, X_test, task)
                # X_train = self.config.sanitize(X_train)
                data, label_mapping, model_classes = self.get_data(X_train, y_train, X_test, task)
                test_preds = np.zeros(len(X_test))

                self.learn = Learner(self.model, self.opt, self.criterion, data, label_mapping, model_classes)
                self.learn.device = self.device
                self.learn.model.to(self.device)
                gc.collect()

                # torch.cuda.empty_cache()
                # torch.clear_autocast_cache()

                run = Runner(cb_funcs=self.cbfs)
                self.learn.model.train()
                run.fit(epochs=self.config.epoch_count, learn = self.learn)
                self.learn.model.eval()

                print("Loss" ,run.loss)

                # for i, (x_batch,) in enumerate(data.test_dl):
                #     with torch.no_grad():
                #         y_pred = learn.model(x_batch).detach()
                #     test_preds_fold[i*self.batch_size:(i+1)*self.batch_size] = self.sigmoid(y_pred.cpu().numpy())[:, 0]

                # batch_size = 4

                # test fold
                
                # for i, (x_batch,) in enumerate(data.test_dl):
                #     with torch.no_grad():
                #         y_pred = learn.model(x_batch).detach()
                    # test_preds[i*batch_size]


                # self.call_callbacks("on_train_begin")
                # for epoch in tqdm(
                #     range(self.config.epoch_count),
                #     disable=disable_progress_bar,
                #     desc="Epochs",
                # ):
                #     if self.training_state == True:
                #         self.epoch = epoch
                #         self.call_callbacks("on_epoch_begin")
                #         correct = 0
                #         incorrect = 0
                #         running_loss = 0.0

                #         self.model.train()

                #         for batch_idx, (inputs, labels) in enumerate(train_loader):
                #             self.call_callbacks("on_batch_begin")
                #             inputs = self.config.sanitize(inputs)

                #             # if torch.cuda.is_available():
                #             inputs = inputs.to(self.config.device)
                #             labels = labels.to(self.config.device)

                #             # Below two lines are hack to convert model to onnx
                #             global sample_input
                #             sample_input = inputs

                #             def _optimizer_step():
                #                 outputs = self.model(inputs)
                #                 self.call_callbacks("on_loss_begin")
                #                 self.loss = self.criterion(outputs, labels)
                #                 self.loss.backward()
                #                 self.call_callbacks("on_loss_end")
                #                 self.opt.zero_grad()
                #                 return self.loss

                #             if labels.dtype != torch.int64:
                #                 labels = torch.tensor(
                #                     labels, dtype=torch.long, device=labels.device
                #                 )

                #             self.call_callbacks("on_opt_step_begin")
                #             self.loss_opt = self.opt.step(_optimizer_step)
                #             self.call_callbacks("on_opt_step_end")

                #             # self.scheduler.step(self.loss_opt)

                #             predicted = self.model(inputs)
                #             predicted = self.config.predict(predicted, task)

                #             accuracy = float("nan")  # type: float
                #             if isinstance(task, OpenMLClassificationTask):
                #                 correct += (predicted == labels).sum()
                #                 incorrect += (predicted != labels).sum()
                #                 accuracy_tensor = (
                #                     torch.tensor(1.0) * correct / (correct + incorrect)
                #                 )
                #                 accuracy = accuracy_tensor.item()

                #             # Print training progress information
                #             running_loss += self.loss_opt.item()
                #             if batch_idx % 100 == 99:  #  print every 100 mini-batches
                #                 print(
                #                     f"Epoch: {epoch + 1}, Batch: {batch_idx + 1:5d}, Loss: {running_loss / 100:.3f}"
                #                 )
                #                 running_loss = 0.0

                #             self.config.progress_callback(
                #                 fold_no,
                #                 rep_no,
                #                 epoch,
                #                 batch_idx,
                #                 self.loss_opt.item(),
                #                 accuracy,
                #             )
                #             self.call_callbacks("on_batch_end")

                #     # validation phase

                #     self.model.eval()
                #     correct_val = 0
                #     incorrect_val = 0
                #     val_loss = 0

                #     with torch.no_grad():
                #         for inputs_val, labels_val in enumerate(val_loader):

                #             # if torch.cuda.is_available():
                #             inputs_val = inputs.to(self.config.device)
                #             labels_val = labels.to(self.config.device)
                #             outputs_val = self.model(inputs_val)
                #             if labels_val.dtype != torch.int64:
                #                 labels_val = torch.tensor(
                #                     labels_val,
                #                     dtype=torch.long,
                #                     device=labels.device,
                #                 )
                #             loss_val = self.criterion(outputs_val, labels_val)

                #             predicted_val = self.config.predict(outputs_val, task)
                #             correct_val += (
                #                 (predicted_val == labels_val).sum().item()
                #             )
                #             incorrect_val += (
                #                 (predicted_val != labels_val).sum().item()
                #             )

                #             val_loss += loss_val.item()

                #     accuracy_val = correct_val / (correct_val + incorrect_val)

                #     # Print validation metrics
                #     print(
                #         f"Epoch: {epoch + 1}, Validation Loss: {val_loss / len(val_loader):.3f}, Validation Accuracy: {accuracy_val:.3f}"
                #     )
                #     self.call_callbacks("on_epoch_end")
                # self.call_callbacks("on_train_end")

        except AttributeError as e:
            # typically happens when training a regressor8 on classification task
            raise PyOpenMLError(str(e))

        # if isinstance(task, OpenMLClassificationTask):
        #     # Convert class labels to numerical indices

        #     x_train_labels = (
        #         X_train_train["encoded_labels"]
        #         if self.config.perform_validation
        #         else (
        #             X_train["Class_encoded"]
        #             if "Class_encoded" in X_train
        #             else X_train["encoded_labels"]
        #         )
        #     )
        #     model_classes = np.sort(x_train_labels.astype("int").unique())
            # model_classes = np.amax(y_train)

        # In supervised learning this returns the predictions for Y
        if isinstance(task, OpenMLSupervisedTask):
            # self.model.eval()

            # name = task.get_dataset().name
            # dataset_name = name.split('Meta_Album_')[1] if 'Meta_Album' in name else name

            # test, _ = self.openml2pytorch_data(X_test, None, task)
            # test_loader = torch.utils.data.DataLoader(
            #     test,
            #     batch_size=self.config.batch_size,
            #     shuffle=False,
            #     pin_memory=self.config.device != "cpu",
            # )
            pred_y = self.pred_test(task, self.model, data.test_dl, self.config.predict)
        else:
            raise ValueError(task)

        if isinstance(task, OpenMLClassificationTask):

            try:
                self.model.eval()

                # test, _ = self.openml2pytorch_data(X_test, None, task)
                # test_loader = torch.utils.data.DataLoader(
                #     test,
                #     batch_size=self.config.batch_size,
                #     shuffle=True,
                #     pin_memory=self.config.device != "cpu",
                # )

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
