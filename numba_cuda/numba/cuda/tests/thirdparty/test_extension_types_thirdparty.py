import unittest
from numba.cuda.testing import CUDATestCase

from importlib.util import find_spec

if not find_spec("cudf"):
    raise unittest.SkipTest("cudf not installed, skipping tests")
else:
    import cudf
    from pandas.testing import assert_series_equal


class TestCudf(CUDATestCase):
    def run(self, result=None):
        super(CUDATestCase, self).run(result)

    def test_cudf_udf_basic(self):
        expect = cudf.Series([2, 4, 6, 8])
        got = cudf.Series([1, 2, 3, 4]).apply(lambda x: x * 2)
        assert_series_equal(expect.to_pandas(), got.to_pandas())

    def test_cudf_udf_string_readonly(self):
        def f(string):
            return len(string) * 2

        expect = cudf.Series([6, 8, 10, 12])
        got = cudf.Series(["abc", "defg", "hijkl", "mnopqr"]).apply(f)
        assert_series_equal(expect.to_pandas(), got.to_pandas())

    def test_cudf_udf_string_intermediate_string(self):
        def f(string):
            result = string + string
            return len(result)

        expect = cudf.Series([6, 8, 10, 12], dtype="int32")
        got = cudf.Series(["abc", "defg", "hijkl", "mnopqr"]).apply(f)
        assert_series_equal(expect.to_pandas(), got.to_pandas())

    def test_cudf_udf_string_return_string(self):
        def f(string):
            return string + string

        expect = cudf.Series(
            ["abcabc", "defgdefg", "hijklhijkl", "mnopqrmnopqr"]
        )
        got = cudf.Series(["abc", "defg", "hijkl", "mnopqr"]).apply(f)
        assert_series_equal(expect.to_pandas(), got.to_pandas())

    def test_cudf_udf_gby_apply(self):
        def f(group):
            return (group["a"].max() - group["a"].min()) / 2

        df = cudf.DataFrame(
            {
                "key": [1, 1, 1, 2, 2, 2],
                "a": [1, 2, 3, 4, 6, 8],
            }
        )

        expect = cudf.Series(
            [1.0, 2.0], index=cudf.Series([1, 2], dtype="int64", name="key")
        )
        got = df.groupby("key").apply(f, engine="jit")

        assert_series_equal(expect.to_pandas(), got.to_pandas())
