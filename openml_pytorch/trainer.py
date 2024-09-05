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
from .data import OpenMLImageDataset
from openml.exceptions import PyOpenMLError
import openml


class OpenMLTrainerModule:

    def __init__(self, model_config, data_config):
        # self.model_config: SimpleNamespace = openml.runs.data_config
        # self.data_config: SimpleNamespace = openml.runs.model_config
        self.model_config = model_config
        self.data_config = data_config
        self.config = SimpleNamespace(**{**self.model_config.__dict__, **self.data_config.__dict__})
        self.user_defined_measures = OrderedDict()

        # self.setup_configs(model_config, data_config)
    
    # def setup_configs(self, model_config, data_config):
    #     self.model_config = model_config
    #     self.data_config = data_config
    
    
    def openml2pytorch_data(self, X, y, task) -> Any:
        # convert openml dataset to pytorch compatible dataset

        df = X
        columns_to_use = [self.config.filename_col]

        if y is not None:
            label_encoder = preprocessing.LabelEncoder().fit(y)
            df.loc[:, 'encoded_labels'] = label_encoder.transform(y)
            label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
            columns_to_use = [self.config.filename_col, 'encoded_labels']
        else:
            label_mapping = None

        def convert_to_rgb(image):
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image

        data = OpenMLImageDataset(
            image_size= self.config.image_size,
            annotations_df= df[columns_to_use],
            img_dir = self.config.file_dir,
            transform = Compose([
                ToPILImage(),  # Convert tensor to PIL Image to ensure PIL Image operations can be applied.
                Lambda(convert_to_rgb),  # Convert PIL Image to RGB if it's not already.
                Resize((self.config.image_size, self.config.image_size)),  # Resize the image.
                ToTensor()  # Convert the PIL Image back to a tensor.
            ])
            )
        
        return data, label_mapping
        
    
    def _prediction_to_probabilities(self,y: np.ndarray, classes: List[Any]) -> np.ndarray:
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
                raise ValueError('please convert model classes to list prior to '
                                 'calling this fn')
            result = np.zeros((len(y), len(classes)), dtype=np.float32)
            for obs, prediction_idx in enumerate(y):
                result[obs][prediction_idx] = 1.0
            return result
    
    def run_model_on_fold(self, model, task, X_train, rep_no, fold_no, y_train, X_test):
        

        model_copy = copy.deepcopy(model).to(self.config.device)
        try:
            if isinstance(task, OpenMLSupervisedTask) or isinstance(task, OpenMLClassificationTask):
                model_copy.train()

                criterion = self.config.criterion_gen(task)
                optimizer = self.config.optimizer_gen(model_copy, task)
                scheduler = self.config.scheduler_gen(optimizer, task)
                pin_memory = False
                
                # if torch.cuda.is_available():
                if self.config.device != "cpu":
                    criterion = criterion.to(self.config.device)
                    pin_memory = True
                
                if self.config.perform_validation:
                    from sklearn.model_selection import train_test_split
                    
                    # TODO: Here we're assuming that X has a label column, this won't work in general
                    X_train_train, x_val, y_train_train, y_val = train_test_split(X_train, y_train, test_size=self.config.validation_split, shuffle=True, stratify=y_train, random_state=0) 
                    train, label_mapping = self.openml2pytorch_data(X_train_train, y_train_train, task)
                    train_loader = torch.utils.data.DataLoader(train, batch_size=self.config.batch_size,
                                                           shuffle=True, pin_memory = pin_memory)
                    
                    val, _ = self.openml2pytorch_data(x_val, None, task)
                    val_loader = torch.utils.data.DataLoader(val, batch_size=self.config.batch_size,
                                                           shuffle=False, pin_memory = pin_memory)
                    
                else:
                    
                    train, label_mapping = self.openml2pytorch_data(X_train, y_train, task)
                    train_loader = torch.utils.data.DataLoader(train, batch_size=self.config.batch_size,
                                                           shuffle=True, pin_memory = pin_memory)
                
                for epoch in range(self.config.epoch_count):
                    correct = 0
                    incorrect = 0
                    running_loss = 0.0

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        inputs = self.config.sanitize(inputs)
                        
                        # if torch.cuda.is_available():
                        inputs = inputs.to(self.config.device)
                        labels = labels.to(self.config.device)
                        
                         # Below two lines are hack to convert model to onnx
                        global sample_input
                        sample_input = inputs
                            
                        def _optimizer_step():
                            optimizer.zero_grad()
                            outputs = model_copy(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            return loss

                        if labels.dtype != torch.int64:
                                labels = torch.tensor(labels, dtype=torch.long, device = labels.device)
                        loss_opt = optimizer.step(_optimizer_step)
                        scheduler.step(loss_opt)

                        predicted = model_copy(inputs)
                        predicted = self.config.predict(predicted, task)

                        accuracy = float('nan')  # type: float
                        if isinstance(task, OpenMLClassificationTask):
                            correct += (predicted == labels).sum()
                            incorrect += (predicted != labels).sum()
                            accuracy_tensor = torch.tensor(1.0) * correct / (correct + incorrect)
                            accuracy = accuracy_tensor.item()
                            
                        # Print training progress information
                        running_loss += loss_opt.item()
                        if batch_idx % 100 == 99: #  print every 100 mini-batches
                            print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1:5d}, Loss: {running_loss / 100:.3f}')
                            running_loss = 0.

                        self.config.progress_callback(fold_no, rep_no, epoch, batch_idx,
                                          loss_opt.item(), accuracy)
                
                    # validation phase
                    if self.config.perform_validation:
                      
                        model_copy.eval()
                        correct_val = 0
                        incorrect_val = 0
                        val_loss = 0
                        
                        with torch.no_grad():
                            for inputs_val, labels_val in enumerate(val_loader):
                                
                                # if torch.cuda.is_available():
                                inputs_val = inputs.to(self.config.device)
                                labels_val = labels.to(self.config.device)
                                outputs_val = model_copy(inputs_val)
                                if labels_val.dtype != torch.int64:
                                    labels_val = torch.tensor(labels_val, dtype=torch.long, device = labels.device)
                                loss_val = criterion(outputs_val, labels_val)
                                
                                predicted_val = self.config.predict(outputs_val, task)
                                correct_val += (predicted_val == labels_val).sum().item()
                                incorrect_val += (predicted_val != labels_val).sum().item()
                        
                                val_loss += loss_val.item()
                                
                        accuracy_val = correct_val/(correct_val + incorrect_val)
                        
                        # Print validation metrics
                        print(f'Epoch: {epoch + 1}, Validation Loss: {val_loss / len(val_loader):.3f}, Validation Accuracy: {accuracy_val:.3f}')
                
        except AttributeError as e:
            # typically happens when training a regressor8 on classification task
            raise PyOpenMLError(str(e))

        if isinstance(task, OpenMLClassificationTask):
            # Convert class labels to numerical indices
            
            x_train_labels = (
                X_train_train['encoded_labels']
                if self.config.perform_validation
                else (X_train['Class_encoded']
                    if 'Class_encoded' in X_train
                    else X_train['encoded_labels'])
                )
            model_classes =  np.sort(x_train_labels.astype('int').unique())
            # model_classes = np.amax(y_train)

        # In supervised learning this returns the predictions for Y
        if isinstance(task, OpenMLSupervisedTask):
            model_copy.eval()
                
            #name = task.get_dataset().name
            #dataset_name = name.split('Meta_Album_')[1] if 'Meta_Album' in name else name
            
            test, _ = self.openml2pytorch_data(X_test, None, task)
            test_loader = torch.utils.data.DataLoader(test, batch_size=self.config.batch_size,
                                                           shuffle=False, pin_memory = self.config.device != 'cpu')
            probabilities = []
            for batch_idx, inputs in enumerate(test_loader):
                inputs = self.config.sanitize(inputs)
                # if torch.cuda.is_available():
                inputs = inputs.to(self.config.device)
                            
                # Perform inference on the batch
                pred_y_batch = model_copy(inputs)
                pred_y_batch = self.config.predict(pred_y_batch, task)
                pred_y_batch = pred_y_batch.cpu().detach().numpy()

                probabilities.append(pred_y_batch)
            
            # Concatenate probabilities from all batches
            pred_y = np.concatenate(probabilities, axis=0)
        else:
            raise ValueError(task)

        if isinstance(task, OpenMLClassificationTask):

            try:
                model_copy.eval()
                
                test, _ = self.openml2pytorch_data(X_test, None, task)
                test_loader = torch.utils.data.DataLoader(test, batch_size=self.config.batch_size,
                                                          shuffle=True, pin_memory = self.config.device != 'cpu')
                
            
                probabilities = []
                for batch_idx, inputs in enumerate(test_loader):
                    inputs = self.config.sanitize(inputs)
                    # if torch.cuda.is_available():
                    inputs = inputs.to(self.config.device)
                    # Perform inference on the batch
                    proba_y_batch = model_copy(inputs)
                    proba_y_batch = self.config.predict_proba(proba_y_batch)
                    proba_y_batch = proba_y_batch.cpu().detach().numpy()

                    probabilities.append(proba_y_batch)
                
                # Concatenate probabilities from all batches
                proba_y = np.concatenate(probabilities, axis=0)
                
            except AttributeError:    
                if task.class_labels is not None:
                    proba_y = self._prediction_to_probabilities(pred_y, list(task.class_labels))
                else:
                    raise ValueError('The task has no class labels')
            
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
                        proba_y.shape[1], len(task.class_labels),
                    )
                    warnings.warn(message)
                    self.config.logger.warning(message)
            else:
                raise ValueError('The task has no class labels')

        elif isinstance(task, OpenMLRegressionTask):
            proba_y = None

        else:
            raise TypeError(type(task))
        
        # Convert model to onnx 
        onnx_ = self._onnx_export(model_copy)
        
        global last_models
        last_models = onnx_
        
        return pred_y, proba_y, self.user_defined_measures, None

    def _onnx_export(self, model_copy):
        f = io.BytesIO()
        torch.onnx.export(model_copy, sample_input, f)
        onnx_model = onnx.load_model_from_string(f.getvalue())
        onnx_ = onnx_model.SerializeToString()
        return onnx_