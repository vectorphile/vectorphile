"""
Example 04: DataFrame & Series — pandas-compatible data manipulation

Demonstrates:
- DataFrame creation and indexing
- GroupBy aggregations
- Merge (join)
- Pivot, melt
- Missing data handling
- Statistical methods
- CSV read/write
"""
import numpy as np
import vectorphile as vp
from vectorphile.data import DataFrame, Series, read_csv
import io

np.random.seed(42)

# ── Create a DataFrame ────────────────────────────────────────────────────────
print("=" * 55)
print("Creating employee DataFrame")
print("=" * 55)

departments = ['Engineering', 'Marketing', 'HR', 'Finance']
names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Helen']
df = DataFrame({
    'name':       names,
    'dept':       [departments[i % 4] for i in range(8)],
    'salary':     np.round(50000 + np.random.randn(8) * 10000, 2),
    'years_exp':  np.random.randint(1, 15, 8).astype(float),
    'rating':     np.round(3.0 + np.random.randn(8) * 0.5, 1).clip(1, 5),
})
print(df)

# ── Basic stats ───────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Describe")
print("=" * 55)
print(df.describe())

# ── Filtering ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Employees with salary > 55000")
print("=" * 55)
senior = df[df['salary'] > 55000]
print(senior[['name', 'dept', 'salary']])

# ── New column ───────────────────────────────────────────────────────────────
df['bonus'] = df['salary'] * (df['rating'] / 5.0) * 0.1
df['seniority'] = df['years_exp'].apply(lambda x: 'Senior' if x >= 7 else 'Junior')

# ── GroupBy ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Average salary by department")
print("=" * 55)
dept_stats = df.groupby('dept').mean()
print(dept_stats[['salary', 'years_exp', 'rating']])

print("\n" + "=" * 55)
print("Employee count by department")
print("=" * 55)
print(df.groupby('dept').count())

# ── Merge / Join ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Merge with performance reviews")
print("=" * 55)
reviews = DataFrame({
    'name':        ['Alice', 'Charlie', 'Eve', 'Grace'],
    'q4_score':    [92, 87, 95, 78],
    'promoted':    [True, False, True, False],
})
merged = df.merge(reviews, on='name', how='left')
print(merged[['name', 'dept', 'salary', 'q4_score', 'promoted']].head(6))

# ── Missing values ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Missing value handling")
print("=" * 55)
df_with_nulls = DataFrame({'a': [1.0, np.nan, 3.0, np.nan], 'b': [4.0, 5.0, np.nan, 7.0]})
print("Before fillna:")
print(df_with_nulls)
print("\nAfter fillna(0):")
print(df_with_nulls.fillna(0.0))
print(f"\nRows with no nulls: {len(df_with_nulls.dropna())}")

# ── Melt ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Melt (wide → long format)")
print("=" * 55)
wide = DataFrame({'id': [1, 2], 'jan': [100.0, 200.0], 'feb': [110.0, 190.0]})
long = wide.melt(id_vars=['id'], value_vars=['jan', 'feb'], var_name='month', value_name='revenue')
print(long)

# ── CSV round-trip ────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("CSV export and re-import")
print("=" * 55)
csv_str = df[['name', 'dept', 'salary']].to_csv(index=False)
df_reimported = read_csv(io.StringIO(csv_str))
print(f"Exported {len(df)} rows, reimported {len(df_reimported)} rows")
print(df_reimported.head(3))

# ── Series operations ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Series operations on salary column")
print("=" * 55)
salaries = df['salary']
print(f"Min:    ${salaries.min():,.2f}")
print(f"Max:    ${salaries.max():,.2f}")
print(f"Mean:   ${salaries.mean():,.2f}")
print(f"Median: ${salaries.quantile(0.5):,.2f}")
print(f"Std:    ${salaries.std():,.2f}")

normalized = (salaries - salaries.mean()) / salaries.std()
print(f"\nNormalized salary stats: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
