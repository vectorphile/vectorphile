"""vectorphile.data.sql — SQL query interface for DataFrames.

Execute SQL-style queries on vectorphile DataFrames using Python's built-in
sqlite3 module as the execution engine.

Usage::

    from vectorphile.data.sql import query, register, sql

    df = vp.data.read_csv('sales.csv')
    result = query('SELECT region, SUM(amount) as total FROM df GROUP BY region', df=df)

    # Register multiple tables
    register('sales', df)
    register('customers', customers_df)
    result = sql('SELECT s.*, c.name FROM sales s JOIN customers c ON s.id = c.id')
"""
from __future__ import annotations
import sqlite3
import io
from typing import Optional, Dict, Any


# Global in-memory database for registered tables
_conn: Optional[sqlite3.Connection] = None
_tables: Dict[str, Any] = {}


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(':memory:')
    return _conn


def _df_to_sqlite(df, table_name: str, conn: sqlite3.Connection) -> None:
    """Write a vectorphile DataFrame to a SQLite table."""
    cols = list(df.columns)
    col_defs = []
    for col in cols:
        try:
            vals = df[col].values if hasattr(df[col], 'values') else list(df[col])
            # Determine type from first non-None value
            sample = next((v for v in vals if v is not None), None)
            if isinstance(sample, int) and not isinstance(sample, bool):
                col_defs.append(f'"{col}" INTEGER')
            elif isinstance(sample, float):
                col_defs.append(f'"{col}" REAL')
            else:
                col_defs.append(f'"{col}" TEXT')
        except Exception:
            col_defs.append(f'"{col}" TEXT')

    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    create_sql = f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})'
    conn.execute(create_sql)

    placeholders = ', '.join(['?'] * len(cols))
    insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
    rows = []
    for i in range(len(df)):
        row = []
        for col in cols:
            try:
                val = df[col][i]
                if hasattr(val, 'item'):
                    val = val.item()
                row.append(val)
            except Exception:
                row.append(None)
        rows.append(tuple(row))
    conn.executemany(insert_sql, rows)
    conn.commit()


def _sqlite_to_df(cursor: sqlite3.Cursor) -> Any:
    """Convert sqlite3 cursor result to a vectorphile DataFrame."""
    try:
        from vectorphile.data.frame import DataFrame
        rows = cursor.fetchall()
        if not rows:
            cols = [desc[0] for desc in (cursor.description or [])]
            return DataFrame({c: [] for c in cols})
        cols = [desc[0] for desc in cursor.description]
        data = {c: [row[i] for row in rows] for i, c in enumerate(cols)}
        return DataFrame(data)
    except Exception:
        rows = cursor.fetchall()
        return rows


def query(sql_str: str, **kwargs) -> Any:
    """Execute a SQL query with DataFrames passed as keyword args.

    Usage::

        result = query('SELECT * FROM df WHERE amount > 100', df=my_df)
        result = query('SELECT a.*, b.val FROM df1 a JOIN df2 b ON a.id = b.id',
                       df1=df1, df2=df2)
    """
    conn = sqlite3.connect(':memory:')
    # Load all provided DataFrames as tables
    for name, df in kwargs.items():
        _df_to_sqlite(df, name, conn)
    cursor = conn.execute(sql_str)
    result = _sqlite_to_df(cursor)
    conn.close()
    return result


def register(name: str, df) -> None:
    """Register a DataFrame as a named SQL table (persists until unregistered)."""
    global _tables
    _tables[name] = df
    conn = _get_conn()
    _df_to_sqlite(df, name, conn)


def unregister(name: str) -> None:
    """Remove a registered table."""
    global _tables
    _tables.pop(name, None)
    conn = _get_conn()
    conn.execute(f'DROP TABLE IF EXISTS "{name}"')
    conn.commit()


def sql(sql_str: str) -> Any:
    """Execute a SQL query on previously registered tables."""
    conn = _get_conn()
    cursor = conn.execute(sql_str)
    return _sqlite_to_df(cursor)


def list_tables() -> list:
    """List currently registered table names."""
    conn = _get_conn()
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]


def read_sql(sql_str: str, con=None) -> Any:
    """Read a SQL query result into a DataFrame.

    Args:
        sql_str: SQL SELECT statement.
        con: sqlite3 connection or None to use the global in-memory DB.
    """
    conn = con or _get_conn()
    cursor = conn.execute(sql_str)
    return _sqlite_to_df(cursor)


__all__ = [
    'query', 'register', 'unregister', 'sql', 'list_tables', 'read_sql',
]
