import pytest
import pandas as pd
import numpy as np
from tanami.timeseries import TimeSeries, split_series


class TestTimeSeries:

    @pytest.fixture(scope='module')
    def ts_data(self):
        return pd.DataFrame(
            data=[
                [7767, 3883, 7, 13],
                [7544, 3749, 7, 14],
                [7869, 3884, 7, 15],
                [7254, 3836, 7, 16]
            ],
            columns=['qty', 'count', 'month', 'day']
        )

    def test_normalized_series_single(self, ts_data):
        ts = TimeSeries(ts_data)
        np.testing.assert_array_almost_equal(ts.normalized_series('qty', method='mean'),
                                             [[0.67156646], [-0.2732873], [1.10374172], [-1.50202089]])

    def test_normalized_series_multiple(self, ts_data):
        ts = TimeSeries(ts_data)
        np.testing.assert_array_almost_equal(ts.normalized_series(['qty', 'count'], method='mean'),
                                             [[0.67156646, 0.819334],
                                              [-0.2732873, -1.62046],
                                              [1.10374172, 0.837541],
                                              [-1.50202089, -0.036415]])

    def test_normalized_series_minmax(self, ts_data):
        ts = TimeSeries(ts_data)
        np.testing.assert_array_almost_equal(ts.normalized_series(['qty', 'count'], method='minmax'),
                                             [[0.834146, 0.992593],
                                              [0.471545, 0],
                                              [1, 1],
                                              [0, 0.644444]])

    def test_sequenced_series(self, ts_data):
        ts = TimeSeries(ts_data)
        X, y = ts.sequenced_series(['qty', 'count'], 'count', step=2)
        np.testing.assert_array_almost_equal(X,
                                             [
                                                 [[0.834146, 0.992593], [0.471545, 0]],
                                                 [[0.471545, 0], [1, 1]]
                                             ])
        np.testing.assert_array_almost_equal(y,
                                             [
                                                 [1.],
                                                 [0.64444444]
                                             ])

    def split_seq_data(self):
        return pd.DataFrame(
            data=[
                [7767, 3883, 7, 13],
                [7544, 3749, 7, 14],
                [7869, 3884, 7, 15],
                [7254, 3836, 7, 16],
                [6462, 3707, 7, 17],
                [7747, 3904, 7, 18],
                [7259, 3697, 7, 19],
                [7829, 3896, 7, 20],
                [7564, 3767, 7, 21],
                [7866, 3893, 7, 22],
                [7271, 3850, 7, 23],
                [6534, 3712, 7, 24],
                [7792, 3921, 7, 25],
                [7280, 3718, 7, 26],
                [7799, 3893, 7, 27],
                [7563, 3746, 7, 28],
                [7862, 3893, 7, 29],
                [7243, 3852, 7, 30],
                [6470, 3708, 7, 31],
                [7712, 3897, 8, 1],
                [7218, 3714, 8, 2],
                [7685, 3876, 8, 3]
            ],
            columns=['qty', 'count', 'month', 'day']
        )


    def test_split_series(self):
        # data size is 22.   Split into 0.5 = 11 each.  Step is 2 so 11 - 2 = 9
        train, test = split_series(self.split_seq_data(), ['qty', 'count'], 'qty', 2, split=0.5)
        assert len(train[0]) == 9
        assert len(test[0]) == 9