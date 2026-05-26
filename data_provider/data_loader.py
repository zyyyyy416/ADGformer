import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


#for Flight 4:4:2 split
class Dataset_Flight(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, timeenc=0, freq='h',
                 numpoint_win=24, w_bias=0):
        """
        自动识别是否存在 'date' 列：
        - 若有 'date' 列，则按原逻辑提取时间特征（data_stamp）
        - 若无 'date' 列，则不使用时间特征（self.data_stamp = None）
        """

        # size: [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.numpoint_win = numpoint_win
        self.w_bias = w_bias
        self.root_path = root_path
        self.data_path = data_path

        # 读取并处理数据
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 自动判断是否存在 'date' 列
        self.has_date = 'date' in df_raw.columns

        # ===== 1. 列重排 / 选择数值列 =====
        if self.has_date:
            # 确保 'date' 在第一列，其他特征在后
            cols = list(df_raw.columns)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]  # 除去 date 的所有特征列
        else:
            # 无 date 列，则所有列都作为数值特征
            cols_data = df_raw.columns

        df_data = df_raw[cols_data]

        # ===== 2. 划分 train/val/test 边界（与是否有 date 无关）=====
        num_total = len(df_raw)
        num_train = int(num_total * 0.4)
        num_test = int(num_total * 0.2)
        num_vali = num_total - num_train - num_test

        border1s = [
            0,
            num_train - self.seq_len,
            num_total - num_test - self.seq_len
        ]
        border2s = [
            num_train,
            num_train + num_vali,
            num_total
        ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # ===== 3. 标准化（仅对数值特征做）=====
        self.size = len(df_data)
        ind = np.arange(0, self.size)

        if self.scale:
            # 用训练集部分拟合 scaler（始终使用 train 段）
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ===== 4. 时间特征（只在有 date 列时才生成）=====
        if self.has_date and self.timeenc in [0, 1]:
            df_stamp = df_raw[['date']][border1:border2].copy()
            df_stamp['date'] = pd.to_datetime(df_stamp['date'])

            if self.timeenc == 0:
                # 简单时间特征
                df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
                df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
                df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
                df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
                data_stamp = df_stamp.drop(['date'], axis=1).values
            else:  # self.timeenc == 1
                # 高级时间编码，需自行确保 time_features 可用
                data_stamp = time_features(
                    pd.to_datetime(df_stamp['date'].values),
                    freq=self.freq
                )
                data_stamp = data_stamp.transpose(1, 0)

            self.data_stamp = data_stamp
            self.has_time_features = True
        else:
            # 没有 date 或不希望使用时间编码时
            self.data_stamp = None
            self.has_time_features = False

        # ===== 5. 当前集合的数据切片（X / Y）=====
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # ===== 6. 时间窗口索引（与 date 无关，只用行号）=====
        self.time_index = ind[border1:border2]
        self.time_index = (self.time_index + self.w_bias) // self.numpoint_win

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_index = self.time_index[s_begin:s_end]

        # 为了完全兼容原来的使用方式，这里仍然只返回 3 个：
        # seq_x, seq_y, seq_x_index
        # 如果以后你要用 time 特征，可以改成一并返回 self.data_stamp 对应片段。
        return seq_x, seq_y, seq_x_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
# 高过
class Dataset_htsuperheater(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='htsuperheater.csv',  # 你自己的文件名
                 scale=True, timeenc=0, freq='t',
                 numpoint_win=60, w_bias=0):
        """
        数据说明：
        - 1 个月数据，每分钟一条
        - 每天 24 小时，每小时 60 条

        参数：
        - size: [seq_len, label_len, pred_len]
        - flag: 'train' / 'val' / 'test'
        - freq: 't' 表示 minute 级别
        """
        # 序列长度设定
        if size is None:
            #
            self.seq_len = 60*2   
            self.label_len = 60     
            self.pred_len = 30      
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # train / val / test
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.numpoint_win = numpoint_win  # 这里保留参数，只是当前不再用它做 time_index 划分
        self.w_bias = w_bias
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features)]
        要求有一列叫 'date'，是时间戳（字符串也可以，后面会转成 datetime）
        """

        # 确保 'date' 在第一列
        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]

        # 按 7:2:1 划分 train/val/test
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [
            0,
            num_train - self.seq_len,                  # val 起点要往前回看 seq_len
            len(df_raw) - num_test - self.seq_len      # test 同理
        ]
        border2s = [
            num_train,
            num_train + num_vali,
            len(df_raw)
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 数值特征部分（去掉 date）
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            # 标准化只用 train 部分拟合，避免泄露
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 时间戳部分
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        # 时间特征（给 Transformer 或其它模块用）
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp['date'].apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:  # timeenc == 1
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values),
                freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)  # [T, d_time]

        # 数值数据切片
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # 🔥 关键修改：动态图用“小时”做 index（0~23），每小时一张图
        hour_of_day = df_stamp['date'].dt.hour.values  # shape: [border2-border1], 值在 0~23
        self.time_index = hour_of_day

    def __getitem__(self, index):
        """
        这里保持和你原来一样的滑动窗口逻辑：
        - seq_x: encoder 输入
        - seq_y: decoder label + 预测段
        - seq_x_index: encoder 输入对应的 time_index（这里是小时 0~23）
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # seq_x = self.data_x[s_begin:s_end]      # [seq_len, C]
        # seq_y = self.data_y[r_begin:r_end]       # [label_len+pred_len, C]
        # seq_x_index = self.time_index[s_begin:s_end]  # [seq_len]
        seq_x = self.data_x[s_begin:s_end].copy()      # [seq_len, C]
        seq_y = self.data_y[r_begin:r_end].copy()       # [label_len+pred_len, C]
        seq_x_index = self.time_index[s_begin:s_end].copy()  # [seq_len]

        return seq_x, seq_y, seq_x_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, timeenc=0, freq='h',
                 numpoint_win=24, w_bias=0):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.numpoint_win = numpoint_win
        self.w_bias = w_bias
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        self.size = len(df_data)
        ind = np.arange(0, self.size)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.time_index = ind[border1:border2]
        self.time_index = (self.time_index + self.w_bias) // self.numpoint_win

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_index = self.time_index[s_begin:s_end]

        return seq_x, seq_y, seq_x_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTm1.csv', scale=True, timeenc=0, freq='t',
                 numpoint_win=96, w_bias=0):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.numpoint_win = numpoint_win
        self.w_bias = w_bias
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        self.size = len(df_data)
        ind = np.arange(0, self.size)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.time_index = ind[border1:border2]
        self.time_index = (self.time_index + self.w_bias) // self.numpoint_win

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_index = self.time_index[s_begin:s_end]

        return seq_x, seq_y, seq_x_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, timeenc=0, freq='h',
                 numpoint_win=24, w_bias=0):
        """
        自动识别是否存在 'date' 列：
        - 若有 'date' 列，则按原逻辑提取时间特征（data_stamp）
        - 若无 'date' 列，则不使用时间特征（self.data_stamp = None）
        """

        # size: [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.numpoint_win = numpoint_win
        self.w_bias = w_bias
        self.root_path = root_path
        self.data_path = data_path

        # 读取并处理数据
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 自动判断是否存在 'date' 列
        self.has_date = 'date' in df_raw.columns

        # ===== 1. 列重排 / 选择数值列 =====
        if self.has_date:
            # 确保 'date' 在第一列，其他特征在后
            cols = list(df_raw.columns)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]  # 除去 date 的所有特征列
        else:
            # 无 date 列，则所有列都作为数值特征
            cols_data = df_raw.columns

        df_data = df_raw[cols_data]

        # ===== 2. 划分 train/val/test 边界（与是否有 date 无关）=====
        num_total = len(df_raw)
        num_train = int(num_total * 0.7)
        num_test = int(num_total * 0.2)
        num_vali = num_total - num_train - num_test

        border1s = [
            0,
            num_train - self.seq_len,
            num_total - num_test - self.seq_len
        ]
        border2s = [
            num_train,
            num_train + num_vali,
            num_total
        ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # ===== 3. 标准化（仅对数值特征做）=====
        self.size = len(df_data)
        ind = np.arange(0, self.size)

        if self.scale:
            # 用训练集部分拟合 scaler（始终使用 train 段）
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ===== 4. 时间特征（只在有 date 列时才生成）=====
        if self.has_date and self.timeenc in [0, 1]:
            df_stamp = df_raw[['date']][border1:border2].copy()
            df_stamp['date'] = pd.to_datetime(df_stamp['date'])

            if self.timeenc == 0:
                # 简单时间特征
                df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
                df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
                df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
                df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
                data_stamp = df_stamp.drop(['date'], axis=1).values
            else:  # self.timeenc == 1
                # 高级时间编码，需自行确保 time_features 可用
                data_stamp = time_features(
                    pd.to_datetime(df_stamp['date'].values),
                    freq=self.freq
                )
                data_stamp = data_stamp.transpose(1, 0)

            self.data_stamp = data_stamp
            self.has_time_features = True
        else:
            # 没有 date 或不希望使用时间编码时
            self.data_stamp = None
            self.has_time_features = False

        # ===== 5. 当前集合的数据切片（X / Y）=====
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # ===== 6. 时间窗口索引（与 date 无关，只用行号）=====
        self.time_index = ind[border1:border2]
        self.time_index = (self.time_index + self.w_bias) // self.numpoint_win

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_index = self.time_index[s_begin:s_end]

        # 为了完全兼容原来的使用方式，这里仍然只返回 3 个：
        # seq_x, seq_y, seq_x_index
        # 如果以后你要用 time 特征，可以改成一并返回 self.data_stamp 对应片段。
        return seq_x, seq_y, seq_x_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, data_path='ETTh1.csv', scale=True, inverse=False, timeenc=0,
                 freq='15min', cols=None, numpoint_win=24, w_bias=0):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.numpoint_win = numpoint_win
        self.w_bias = w_bias
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        self.size = len(df_data)
        ind = np.arange(0, self.size)

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.time_index = ind[border1:border2]
        self.time_index = (self.time_index + self.w_bias) // self.numpoint_win

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_index = self.time_index[s_begin:s_end]

        return seq_x, seq_y, seq_x_index

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
