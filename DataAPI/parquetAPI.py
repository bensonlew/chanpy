import os

from Common.CEnum import DATA_FIELD, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Common.func_util import str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi
import pandas as pd


def create_item_dict(data, column_name):
    for i in range(len(data)):
        data[i] = parse_time_column(data[i]) if column_name[i] == DATA_FIELD.FIELD_TIME else str2float(data[i])
    return dict(zip(column_name, data))


def parse_time_column(inp):
    # 20210902113000000
    # 2021-09-13
    # convert Timestamp to CTime


    year = inp.year
    month = inp.month
    day = inp.day
    hour = inp.hour
    minute = inp.minute


    return CTime(year, month, day, hour, minute)


# import pyarrow.parquet as pq


class Parquet_API(CCommonStockApi):
    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=None):
        self.columns = [
            DATA_FIELD.FIELD_TIME,
            DATA_FIELD.FIELD_OPEN,
            DATA_FIELD.FIELD_HIGH,
            DATA_FIELD.FIELD_LOW,
            DATA_FIELD.FIELD_CLOSE,
            # DATA_FIELD.FIELD_VOLUME,
            # DATA_FIELD.FIELD_TURNOVER,
            # DATA_FIELD.FIELD_TURNRATE,
        ]  # 每一列字段
        self.time_column_idx = self.columns.index(DATA_FIELD.FIELD_TIME)
        super(Parquet_API, self).__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        fdata_dir = os.environ.get('FDATA', '/home/liubinxu/work/finance_learning/test')
        file_path = f"{fdata_dir}/{self.code}.qfq.parquet"
        if not os.path.exists(file_path):
            raise CChanException(f"file not exist: {file_path}", ErrCode.SRC_DATA_NOT_FOUND)

        table = pd.read_parquet(file_path)
        print("table: ", len(table))
        for row in table.iterrows():
            data_dict = dict(row[1])
            # if len(data) != len(self.columns):
            #     raise CChanException(f"file format error: {file_path}", ErrCode.SRC_DATA_FORMAT_ERROR)

            data = [
                data_dict["datetime"],
                data_dict[DATA_FIELD.FIELD_OPEN],
                data_dict[DATA_FIELD.FIELD_HIGH],
                data_dict[DATA_FIELD.FIELD_LOW],
                data_dict[DATA_FIELD.FIELD_CLOSE],
            ]
            if self.begin_date is not None and str(data[self.time_column_idx]) < self.begin_date:
                continue
            if self.end_date is not None and str(data[self.time_column_idx]) > self.end_date:
                continue
            yield CKLine_Unit(create_item_dict(data, self.columns))

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass
