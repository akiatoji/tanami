import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn import preprocessing

class TimeSeries():

    def __init__(self, df):
        self.df = df.copy()

    def normalized_series(self, cols, method = 'minmax'):
        assert method in ['mean', 'minmax'], "method is not one of mean, minmax"

        if type(cols) == str:
            cols = [cols]

        normalized_columns = list()
        for col_name in cols:
            assert is_numeric_dtype(self.df[col_name]), "%s is a non-numeric column" % col_name

            col_vals = self.df[col_name].values
            col_vals = col_vals.reshape(-1, 1)
            if method == 'mean':
                normalized_col = preprocessing.StandardScaler().fit_transform(col_vals)
            elif method == 'minmax':
                normalized_col = preprocessing.MinMaxScaler().fit_transform(col_vals)

            normalized_columns.append(normalized_col)

        return np.hstack(tuple(normalized_columns))


    def sequenced_series(self, X_cols, y_col, step):
        assert step > 1

        x_seq = self.normalized_series(X_cols)
        stack_len = len(x_seq)

        y_seq = self.normalized_series(y_col)

        X, y = list(), list()

        # now split patterns
        for i in range(stack_len):
            # find the end of this pattern
            end_ix = i + step
            # check if we are beyond the sequence
            if end_ix > stack_len - 1:
                break

            # gather input and output parts of the pattern
            seq_x, seq_y = x_seq[i:end_ix], y_seq[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], len(X_cols)))

        return X, np.array(y)


def split_series(df, X_cols, y_col, step, split=0.8):
    length = len(df.index)
    split_idx = int(length * split)
    X_df = df[:split_idx]
    y_df = df[split_idx:]

    X_seq = TimeSeries(X_df).sequenced_series(X_cols, y_col, step)
    y_seq = TimeSeries(y_df).sequenced_series(X_cols, y_col, step)

    return X_seq, y_seq
