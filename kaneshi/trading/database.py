import pandas as pd
import sqlalchemy as db

from kaneshi.config import ROOT_DIR

from typing import NoReturn


class DataBase:
    def __init__(self, label: str):
        self.db_path = fr'{ROOT_DIR}/trading/databases/' + label
        self.engine = db.create_engine(f'sqlite:///{self.db_path}.db')

    def save_data(self, table_name: str, data: pd.DataFrame) -> NoReturn:
        """ Save dataframe to database """
        data.to_sql(table_name, self.engine, index=False, if_exists='append')

    def read_table(self, table_name: str) -> pd.DataFrame:
        """ Get table from database by table name """
        return pd.read_sql(table_name, self.engine)
