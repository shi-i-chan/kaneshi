import os
import shutil
import pandas as pd
from functools import wraps
from signal import signal, SIGINT

from kaneshi.config import DEF_DF_EXTENSION

from typing import Callable, NoReturn, List


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

