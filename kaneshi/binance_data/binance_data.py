import os
import zipfile
import pandas as pd
from tqdm import tqdm
import urllib.request
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from kaneshi.utils import TempFolderContext, if_file, save_df, read_df
from kaneshi.config import DEF_DF_EXTENSION, DEF_ROUND, DEF_MARKET_DATA_PATH

from typing import List, NoReturn, Tuple, Callable


class DefaultBinanceData:
    def __init__(self,
                 s_date: Tuple[int],
                 e_date: Tuple[int],
                 symbols: List[str],
                 interval: str = '1m',
                 temp_folder: str = 'temp_folder/',
                 ):
        self.s_date = datetime(*s_date)
        self.e_date = datetime(*e_date)
        self.symbols = symbols
        self.interval = interval
        self.temp_folder = temp_folder
        self._current_symbol = None
        self._current_df_fn = None
        self._candles_columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                 'Close time', 'Quote asset volume', 'Number of trades',
                                 'Taker buy base asset volume', 'Taker buy quote asset volume',
                                 'Ignore']

    def _get_current_symbol(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _clear_candle_df(df: pd.DataFrame) -> pd.DataFrame:
        """ Clear raw candles df. Set dt indices and remove trash """
        df['Open time'] = df['Open time'].apply(lambda x: datetime.fromtimestamp(x / 1e3))
        df.index = df['Open time'].values
        df = df.drop(columns=['Open time', 'Ignore', 'Close time']).round(DEF_ROUND)
        return df

    def get_candles(self) -> NoReturn:
        """ Iterate symbols and get data """
        for current_symbol in self.symbols:
            self._current_symbol = current_symbol
            self._current_df_fn = fr'{DEF_MARKET_DATA_PATH}\raw_{self._current_symbol}_{self.interval}'
            self._get_current_symbol(check_filename=f'{self._current_df_fn}.{DEF_DF_EXTENSION}')


class BinanceVisionData(DefaultBinanceData, TempFolderContext):
    def __init__(self,
                 type_: str = 'daily',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type_ = type_

    @staticmethod
    @if_file
    def _parse_data(url, filename: str, *args, **kwargs) -> NoReturn:  # NOQA
        """ Parse data by url and save in filename """
        urllib.request.urlretrieve(url, filename)  # NOQA

    @if_file
    def _unpack_data(self, filename: str, *args, **kwargs) -> NoReturn:  # NOQA
        """ Unpack .zip archive and remove it """
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(self.temp_folder)
        os.remove(filename)

    def _get_candle_url_fn_list(self, day: datetime) -> Tuple[str, str]:
        """ Generate url and filename for specific year and month """
        def_fn = f"{self._current_symbol}-{self.interval}-{day.year}-{'{:02d}'.format(day.month)}"
        fn = def_fn + '.zip' if self.type_ == 'monthly' else def_fn + f"-{'{:02d}'.format(day.day)}.zip"
        link = f'https://data.binance.vision/data/spot/{self.type_}/klines/{self._current_symbol}/{self.interval}/{fn}'
        return link, (self.temp_folder + fn)

    def _get_url_fn_list(self, url_fn_generator: Callable) -> List[Tuple[str, str]]:
        """ Generate urls and filenames for all years and month in given ranges """
        if self.type_ == 'daily':
            n_day = (self.e_date - self.s_date).days + 1
            dates = [self.s_date + timedelta(days=i) for i in range(n_day)]
        else:
            n_month = (self.e_date.year - self.s_date.year) * 12 + self.e_date.month - self.s_date.month + 1
            dates = [self.s_date + relativedelta(months=i) for i in range(n_month)]
        return [url_fn_generator(day) for day in dates]

    def _download_unpack(self, all_url_fn: List) -> NoReturn:
        """ Download and unpack archives """
        fn_list = []
        for url, filename in tqdm(all_url_fn, desc=f'{self._current_symbol}'):
            self._parse_data(url, filename, check_filename=filename)
            fn_list.append(fn := os.path.splitext(filename)[0] + '.csv')
            self._unpack_data(filename, check_filename=fn)
        return fn_list

    def _collect_files(self, fn_list: List) -> NoReturn:
        """ Finally collect all unpacked .csv files """
        dfs = [read_df(os.path.splitext(fn)[0], extension='csv',
                       columns=self._candles_columns, index_col=None) for fn in fn_list]
        full_df = pd.concat(dfs).drop_duplicates()
        full_df = self._clear_candle_df(full_df)
        save_df(full_df, self._current_df_fn)

    @if_file
    def _get_current_symbol(self, *args, **kwargs) -> NoReturn:  # NOQA
        """ Get candles data for specific symbol if it doesn't exist yet """
        url_fn_list = self._get_url_fn_list(self._get_candle_url_fn_list)
        fn_list = self._download_unpack(url_fn_list)
        self._collect_files(fn_list)
