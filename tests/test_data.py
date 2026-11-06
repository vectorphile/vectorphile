"""Tests for DataFrame and Series."""
import numpy as np
import pytest
from vectorphile.data import DataFrame, Series, read_csv
import io


class TestSeries:
    def test_creation(self):
        s = Series([1, 2, 3, 4, 5])
        assert len(s) == 5

    def test_dict_creation(self):
        s = Series({'a': 1, 'b': 2, 'c': 3})
        assert s['a'] == 1
        assert s['b'] == 2

    def test_arithmetic(self):
        s = Series([1.0, 2.0, 3.0])
        result = s + 1
        assert np.allclose(result.values, [2.0, 3.0, 4.0])

    def test_mul(self):
        s = Series([2.0, 3.0])
        assert np.allclose((s * 2).values, [4.0, 6.0])

    def test_comparison(self):
        s = Series([1.0, 2.0, 3.0])
        mask = (s > 1.5)
        assert list(mask.values) == [False, True, True]

    def test_sum_mean(self):
        s = Series([1.0, 2.0, 3.0, 4.0])
        assert s.sum() == 10.0
        assert s.mean() == 2.5

    def test_std(self):
        s = Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(s.std() - np.std([1, 2, 3, 4, 5], ddof=1)) < 1e-6

    def test_apply(self):
        s = Series([1.0, 2.0, 3.0])
        result = s.apply(lambda x: x ** 2)
        assert np.allclose(result.values, [1.0, 4.0, 9.0])

    def test_value_counts(self):
        s = Series(['a', 'b', 'a', 'c', 'b', 'a'])
        vc = s.value_counts()
        assert vc['a'] == 3
        assert vc['b'] == 2

    def test_fillna(self):
        s = Series([1.0, np.nan, 3.0])
        filled = s.fillna(0.0)
        assert filled.values[1] == 0.0

    def test_sort_values(self):
        s = Series([3, 1, 4, 1, 5])
        sorted_s = s.sort_values()
        assert list(sorted_s.values) == [1, 1, 3, 4, 5]

    def test_describe(self):
        s = Series([1.0, 2.0, 3.0, 4.0, 5.0])
        d = s.describe()
        assert d['count'] == 5.0
        assert d['mean'] == 3.0

    def test_head_tail(self):
        s = Series(range(10))
        assert len(s.head(3)) == 3
        assert len(s.tail(3)) == 3

    def test_unique(self):
        s = Series([1, 2, 2, 3, 3, 3])
        assert len(s.unique()) == 3

    def test_nunique(self):
        s = Series(['a', 'b', 'a'])
        assert s.nunique() == 2


class TestDataFrame:
    def _make_df(self):
        return DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'age': [25, 30, 35, 40],
            'salary': [50000.0, 60000.0, 70000.0, 80000.0],
            'dept': ['Engineering', 'Marketing', 'Engineering', 'HR'],
        })

    def test_creation(self):
        df = self._make_df()
        assert df.shape == (4, 4)

    def test_column_access(self):
        df = self._make_df()
        assert len(df['age']) == 4

    def test_multi_column_access(self):
        df = self._make_df()
        sub = df[['age', 'salary']]
        assert sub.shape == (4, 2)

    def test_boolean_indexing(self):
        df = self._make_df()
        mask = df['age'] > 30
        sub = df[mask]
        assert len(sub) == 2

    def test_assign_new_column(self):
        df = self._make_df()
        df['bonus'] = df['salary'] * 0.1
        assert 'bonus' in df.columns

    def test_head(self):
        df = self._make_df()
        assert len(df.head(2)) == 2

    def test_tail(self):
        df = self._make_df()
        assert len(df.tail(2)) == 2

    def test_shape(self):
        df = self._make_df()
        assert df.shape == (4, 4)

    def test_len(self):
        df = self._make_df()
        assert len(df) == 4

    def test_dropna(self):
        df = DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        dropped = df.dropna()
        assert len(dropped) == 1

    def test_fillna(self):
        df = DataFrame({'a': [1.0, np.nan, 3.0]})
        filled = df.fillna(0.0)
        assert filled['a'].values[1] == 0.0

    def test_rename(self):
        df = self._make_df()
        df2 = df.rename(columns={'age': 'years'})
        assert 'years' in df2.columns

    def test_drop_column(self):
        df = self._make_df()
        df2 = df.drop(columns=['dept'])
        assert 'dept' not in df2.columns
        assert df2.shape[1] == 3

    def test_sort_values(self):
        df = self._make_df()
        sorted_df = df.sort_values('age', ascending=False)
        ages = list(sorted_df['age'].values)
        assert ages == sorted(ages, reverse=True)

    def test_reset_index(self):
        df = self._make_df()
        df2 = df.reset_index(drop=True)
        assert df2.index == list(range(4))

    def test_groupby_mean(self):
        df = self._make_df()
        grouped = df.groupby('dept').mean()
        assert len(grouped) == 3  # Engineering, Marketing, HR

    def test_groupby_sum(self):
        df = DataFrame({'dept': ['a', 'a', 'b'], 'val': [1.0, 2.0, 3.0]})
        grouped = df.groupby('dept').sum()
        assert len(grouped) == 2

    def test_merge_inner(self):
        df1 = DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        df2 = DataFrame({'id': [2, 3, 4], 'score': [90, 85, 88]})
        merged = df1.merge(df2, on='id', how='inner')
        assert len(merged) == 2

    def test_to_numpy(self):
        df = DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        arr = df.to_numpy()
        assert arr.shape == (2, 2)

    def test_to_dict(self):
        df = DataFrame({'a': [1, 2], 'b': [3, 4]})
        d = df.to_dict()
        assert 'a' in d
        assert 'b' in d

    def test_to_csv_round_trip(self):
        df = DataFrame({'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]})
        csv_str = df.to_csv(index=False)
        df2 = read_csv(io.StringIO(csv_str))
        assert df2.shape == (3, 2)

    def test_describe(self):
        df = DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        desc = df.describe()
        assert 'count' in desc.index

    def test_concat(self):
        df1 = DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = DataFrame({'a': [5, 6], 'b': [7, 8]})
        result = DataFrame.concat([df1, df2], ignore_index=True)
        assert len(result) == 4

    def test_melt(self):
        df = DataFrame({'id': [1, 2], 'x': [10, 20], 'y': [30, 40]})
        melted = df.melt(id_vars=['id'], value_vars=['x', 'y'])
        assert len(melted) == 4

    def test_iloc(self):
        df = self._make_df()
        row = df.iloc[0]
        assert isinstance(row, DataFrame)
        assert len(row) == 1

    def test_loc(self):
        df = self._make_df()
        sub = df.loc[0:1]
        assert len(sub) == 2

    def test_contains(self):
        df = self._make_df()
        assert 'age' in df
        assert 'unknown' not in df

    def test_iter_columns(self):
        df = self._make_df()
        cols = list(df)
        assert cols == df.columns
