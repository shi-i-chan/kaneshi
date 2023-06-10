import keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from typing import NoReturn
from numpy.typing import NDArray


class ModelMixin:
    dataset: dict
    model: None
    get_predictions: None

    @staticmethod
    def plot_confusion(y_true: NDArray, y_pred: NDArray):
        matrix = confusion_matrix(y_true, y_pred)
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in
                        matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in
                             matrix.flatten() / np.sum(matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                  zip(group_names,
                      group_counts,
                      group_percentages)]
        plt.figure(figsize=(8, 8))
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues')
        plt.show()

    @staticmethod
    def get_metrics(y_true: NDArray, y_pred: NDArray) -> NoReturn:
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print('Accuracy', accuracy)
        print('Recall ', recall)
        print('Precision ', precision)
        print('F1 ', f1)

    def plot_precision_recall_curve(self, kind: str = 'test') -> NoReturn:
        model_probs = self.model.predict(self.dataset[f'x_{kind}'])
        yhat = self.get_predictions(kind=kind)
        model_precision, model_recall, _ = precision_recall_curve(self.dataset[f'y_{kind}'], model_probs)
        model_f1, model_auc = f1_score(self.dataset[f'y_{kind}'], yhat), auc(model_recall, model_precision)

        print('Logistic: f1=%.3f auc=%.3f' % (model_f1, model_auc))
        no_skill = len(self.dataset[f'y_{kind}'][self.dataset[f'y_{kind}'] == 1]) / len(self.dataset[f'y_{kind}'])

        plt.figure(figsize=(20, 10))
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(model_recall, model_precision, marker='.', label='Logistic')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()

    def plot_roc_auc(self, kind: str = 'test') -> NoReturn:
        ns_probs = [0 for _ in range(len(self.dataset[f'y_{kind}']))]
        model_probs = self.model.predict(self.dataset[f'x_{kind}'])

        ns_auc = roc_auc_score(self.dataset[f'y_{kind}'], ns_probs)
        model_auc = roc_auc_score(self.dataset[f'y_{kind}'], model_probs)

        print('No Skill: ROC AUC=%.3f' % ns_auc)
        print('Logistic: ROC AUC=%.3f' % model_auc)

        ns_fpr, ns_tpr, _ = roc_curve(self.dataset[f'y_{kind}'], ns_probs)
        model_fpr, model_tpr, _ = roc_curve(self.dataset[f'y_{kind}'], model_probs)

        plt.figure(figsize=(20, 10))
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(model_fpr, model_tpr, marker='.', label='Logistic')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()


class Model(ModelMixin):
    def __init__(self,
                 dataset=None,
                 model=None,
                 optimizer=None,
                 loss=None,
                 metrics=None,
                 callbacks=None,
                 join_history=True):
        self.dataset = dataset
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = list(metrics) if not isinstance(metrics, list) else metrics
        self.callbacks = list(callbacks) if not isinstance(callbacks, list) else callbacks
        self.history = None
        self.join_history = join_history

    @classmethod
    def from_fn(cls, model_fn: str, *args, **kwargs):
        model = keras.models.load_model(model_fn)
        return cls(model=model, *args, **kwargs)

    def compile(self) -> NoReturn:
        """ Compile model """
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )

    def fit(self, epoch: int, batch_size: int, verbose: int = 1) -> NoReturn:
        """ Fit model """
        history = self.model.fit(
            self.dataset['x_train'],
            self.dataset['y_train'],
            batch_size=batch_size,
            validation_data=(self.dataset['x_val'],
                             self.dataset['y_val']),
            epochs=epoch,
            callbacks=self.callbacks,
            verbose=verbose,
        )
        if self.join_history and self.history is not None:
            for (key, exist_values), new_values in zip(self.history.history.items(), history.history.values()):
                self.history.history[key] = exist_values + new_values
        else:
            self.history = history

    def evaluate(self, verbose: int = 0) -> NoReturn:
        """ Evaluate model on available data """
        ret = {}
        for kind in ['testing', 'train', 'val', 'test']:
            if f'x_{kind}' in self.dataset and f'y_{kind}' in self.dataset:
                ret[kind] = self.model.evaluate(self.dataset[f'x_{kind}'], self.dataset[f'y_{kind}'], verbose=verbose)
        return ret

    def plot_history(self) -> NoReturn:
        """ Show history chart """
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
        colors = ['blue', 'orange', 'green', 'purple', 'black']
        for idx, metric in enumerate(self.metrics):
            metric = metric if isinstance(metric, str) else metric.name
            ax[0].plot(self.history.history[metric], label=str(f'{metric}_train'), color=colors[idx])
            ax[0].plot(self.history.history[f'val_{metric}'], label=str(f'{metric}_test'), color=colors[idx],
                       linestyle='--')
        ax[1].plot(self.history.history['loss'], label='Train loss', color='blue')
        ax[1].plot(self.history.history['val_loss'], label='Val loss', color='orange')
        plt.tight_layout()
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        plt.show()

    def show_confusion(self, kind: str = 'test') -> NoReturn:
        """ Show confusion matrix and metrics """
        y_pred = self.get_predictions(kind=kind)
        y_true = self.dataset[f'y_{kind}']
        self.get_metrics(y_true, y_pred)
        self.plot_confusion(y_true, y_pred)

    def get_predictions(self, kind: str = 'test'):
        y_prob = self.model.predict(self.dataset[f'x_{kind}'])
        if self.loss == 'binary_crossentropy':
            y_pred = ((y_prob > 0.5) + 0).ravel()
        else:
            y_pred = y_prob.argmax(axis=-1)
        return y_pred

    def get_good_indices(self):
        y_classes = self.get_predictions()
        good_indices = np.argwhere(y_classes == 0)
        return good_indices

