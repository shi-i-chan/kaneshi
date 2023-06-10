import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from kaneshi import utils, config

from typing import NoReturn
from numpy.typing import NDArray

sns.set()


class Dataset:
    def __init__(self, raw_dataset=None, ds_fn: str = ''):
        self.raw_dataset = raw_dataset
        self.ds_fn = ds_fn
        self.dataset = {'x_testing': raw_dataset['examples'], 'y_testing': raw_dataset['labels']}

    @classmethod
    def from_fn(cls, ds_fn: str) -> 'Dataset':
        ds_fn = f'{config.DEF_DATASETS_PATH}/{ds_fn}.h5'
        return cls(raw_dataset=utils.ds_from_h5(ds_fn), ds_fn=ds_fn)

    def normalize_each(self) -> NoReturn:
        """ Normalize each element in dataset """
        scaler = MinMaxScaler()
        for kind in ['testing', 'train', 'val', 'test']:
            if f'x_{kind}' in self.dataset and f'y_{kind}' in self.dataset:
                self.dataset[f'x_{kind}'] = np.array(list(map(lambda x: scaler.fit_transform(x),
                                                              self.dataset[f'x_{kind}'])))
        return self

    def make_equal(self) -> NoReturn:
        """ Make classes equal """
        for kind in ['testing', 'train', 'val', 'test']:
            if f'x_{kind}' in self.dataset and f'y_{kind}' in self.dataset:
                examples = self.dataset[f'x_{kind}']
                labels = self.dataset[f'y_{kind}']

                bad_mask = labels == 0
                good_mask = labels == 1
                all_good_examples = examples[good_mask]
                all_bad_examples = examples[bad_mask]

                min_number = min(all_good_examples.shape[0], all_bad_examples.shape[0])
                selected_good_examples = self.__random_sample(all_good_examples, min_number)
                selected_bad_examples = self.__random_sample(all_bad_examples, min_number)

                examples = np.concatenate((selected_good_examples, selected_bad_examples))
                labels = np.concatenate((np.full(selected_good_examples.shape[0], 1),
                                         np.full(selected_bad_examples.shape[0], 0)))
                p = np.random.permutation(len(examples))
                self.dataset[f'x_{kind}'] = examples[p]
                self.dataset[f'y_{kind}'] = labels[p]
        return self

    def flatten(self) -> NoReturn:
        """ Flatten dataset examples """
        for key, value in self.dataset.items():
            if len(value.shape) > 2:
                self.dataset[key] = value.reshape(value.shape[0], np.prod(value.shape[1:]))
        return self

    def get_dataset(self, kind='split'):
        """ Get dataset as arrays """
        if kind == 'all':
            x = np.concatenate((self.dataset['x_train'], self.dataset['x_val'], self.dataset['x_test']))
            y = np.concatenate((self.dataset['y_train'], self.dataset['y_val'], self.dataset['y_test']))
            return x, y
        return self.dataset['x_train'], self.dataset['x_val'], self.dataset['x_test'], \
               self.dataset['y_train'], self.dataset['y_val'], self.dataset['y_test']  # NOQA

    def get_good_bad(self):
        """ Get good and bad examples """
        labels = self.dataset['labels']
        good = self.dataset['examples'][labels == 1]
        bad = self.dataset['examples'][labels == 0]
        return good, bad

    def show_info(self) -> NoReturn:
        """ Show dataset shapes """
        for kind in ['testing', 'train', 'val', 'test']:
            if f'x_{kind}' in self.dataset:
                print(kind, self.dataset[f'x_{kind}'].shape)

    def to_categorical(self):
        for kind in ['testing', 'train', 'val', 'test']:
            if f'y_{kind}' in self.dataset:
                self.dataset[f'y_{kind}'] = keras.utils.to_categorical(self.dataset[f'y_{kind}'])

    def split(self, tt_percent: float = 0.2, tvt_percent: float = 0.6, kind: str = 'tvt') -> NoReturn:
        """ Split dataset to train/val/test """
        ret = {}
        x_train, x_test, y_train, y_test = train_test_split(self.dataset['x_testing'],
                                                            self.dataset['y_testing'],
                                                            test_size=tt_percent,
                                                            shuffle=True)
        if kind == 'tvt':
            x_test, x_val, y_test, y_val = train_test_split(x_test,
                                                            y_test,
                                                            test_size=tvt_percent,
                                                            shuffle=True)
            ret['x_val'], ret['y_val'] = x_val, y_val
        self.dataset = {**ret, **{'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}}
        return self

    def show_random(self) -> NoReturn:
        """ Plot random x_train element """
        import plotly.express as px
        fig = px.imshow(self.dataset['x_train'][0])
        fig.layout.coloraxis.showscale = False
        fig.update_layout(
            height=3000,
            width=300,
        )
        fig.show()

    def plot_classes(self, n_examples: int = 50, col: int = 0) -> NoReturn:
        """ Plot random examples train data """
        old_true_mask = self.dataset['y_test'] == 1
        old_false_mask = self.dataset['y_test'] == 0
        plt.figure(figsize=(16, 8))
        for i in range(n_examples):
            plt.plot(self.dataset['x_test'][old_true_mask][i][:, col], color='g')
            plt.plot(self.dataset['x_test'][old_false_mask][i][:, col], color='r')
        plt.show()

    def plot_random(self, market_df: pd.DataFrame, percent: float = 0.01) -> NoReturn:
        """ Plot random train example with real market data and stops """
        date_indices = pd.to_datetime(self.raw_dataset['indices'])
        idx = np.random.randint(len(date_indices))
        example_close = self.raw_dataset['examples'][idx]
        date_idx = date_indices[idx]
        start_index = date_idx - np.timedelta64(100 - 1, 'm')
        date_idx = date_idx + np.timedelta64(400, 'm')
        market_close = market_df.loc[start_index: date_idx]['Close'].values

        print("Label ", self.raw_dataset['labels'][idx])

        plt.figure(figsize=(20, 10))
        plt.plot(market_close, label='market')
        plt.plot(example_close, label='dataset')
        plt.legend()

        buy_price = example_close[-1]
        stop_loss = buy_price * (1 - percent)
        take_profit = buy_price * (1 + percent)
        plt.axhline(stop_loss, color='r')
        plt.axhline(take_profit, color='g')
        plt.show()

    @staticmethod
    def __random_sample(array: NDArray, num_samples: int) -> NDArray:
        """ Sample n random elements from array """
        idx = np.random.choice(np.arange(array.shape[0]), size=num_samples)
        return array[idx]