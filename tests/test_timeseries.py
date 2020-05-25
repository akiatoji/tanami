import pytest
import pandas as pd
import numpy as np
from tanami.timeseries import TimeSeries


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
        print(y)
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
