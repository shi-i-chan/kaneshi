import io
import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from functools import wraps
import matplotlib.pyplot as plt
from signal import signal, SIGINT
import plotly.graph_objects as go

from kaneshi.config import DEF_DF_EXTENSION

from numpy.typing import NDArray
from typing import Callable, NoReturn, List, Union


def save_df(df: pd.DataFrame, df_fn: str, extension: str = DEF_DF_EXTENSION) -> NoReturn:
    """ Save dataframe with specific type """
    if extension == 'h5':
        df.to_hdf(df_fn + '.h5', key='df', mode='w')
    elif extension == 'csv':
        df.to_csv(df_fn + '.csv')


def read_df(df_fn: str, extension: str = DEF_DF_EXTENSION, columns: List[str] = None, index_col=0) -> pd.DataFrame:
    """ Read dataframe with specific type """
    if extension == 'h5':
        return pd.DataFrame(pd.read_hdf(df_fn + '.h5', key='df', columns=columns))
    elif extension == 'csv':
        return pd.read_csv(df_fn + '.csv', names=columns, index_col=index_col)


def if_file(function: Callable) -> Callable:
    """ Call function only if check_filename not exist """
    @wraps(function)
    def wrapper(*args, **kwargs):
        if 'check_filename' in kwargs:
            if not os.path.exists(kwargs['check_filename']):
                function(*args, **kwargs)
            else:
                print(f'{kwargs["check_filename"]} already exists.')
        elif 'check_filename' not in kwargs:
            function(*args, **kwargs)
    return wrapper


class TempFolderContext:
    def __init__(self, temp_folder: str = 'temp_folder/'):
        self.temp_folder = temp_folder

    def __enter__(self):
        """ Create temp folder """
        signal(SIGINT, self._sigint_handler)
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Remove temp folder """
        shutil.rmtree(self.temp_folder)

    def _sigint_handler(self, signal_received, frame):  # NOQA
        """ Call __exit__ if KeyboardInterrupt """
        self.__exit__(None, None, None)


def process_fig(figure: go.Figure, fig_config: dict, type_: str) -> Union[NoReturn, NDArray[float]]:
    if type_ == 'plotly':
        figure.show(config=fig_config)
    elif type_ == 'plt':
        plot_plotly_as_plt(figure)
    elif type_ == 'array':
        return plotly_to_array(figure)


def plot_plotly_as_plt(figure: go.Figure) -> NoReturn:
    """ Plot figure with matplotlib """
    fig_array = plotly_to_array(figure)
    plt.figure(figsize=(20, 10))
    plt.imshow(fig_array)
    plt.axis('off')
    plt.show()


def plotly_to_array(figure: go.Figure) -> NDArray[float]:
    """ Convert plotly figure to numpy array """
    fig_bytes = figure.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return (np.asarray(img) / 255).astype('float32')  # NOQA

