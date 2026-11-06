from vectorphile.data.series import Series
from vectorphile.data.frame import DataFrame, read_csv, read_json
from vectorphile.data.index import Index, RangeIndex
from vectorphile.data.groupby import GroupBy


def read_parquet(path) -> DataFrame:
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        return DataFrame(table.to_pydict())
    except ImportError:
        raise ImportError("read_parquet requires pyarrow: pip install pyarrow")


def concat(dfs, axis=0, ignore_index=False) -> DataFrame:
    return DataFrame.concat(dfs, axis=axis, ignore_index=ignore_index)


__all__ = [
    'Series', 'DataFrame', 'Index', 'RangeIndex', 'GroupBy',
    'read_csv', 'read_json', 'read_parquet', 'concat',
]
