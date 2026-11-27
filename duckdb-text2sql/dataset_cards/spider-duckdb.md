---
license: cc-by-sa-4.0
language:
- en
task_categories:
- text-generation
- table-question-answering
tags:
- text-to-sql
- sql
- duckdb
- spider
pretty_name: Spider-DuckDB
size_categories:
- 1K<n<10K
source_datasets:
- xlangai/spider
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
---

# Spider-DuckDB

The [Spider](https://yale-lily.github.io/spider) text-to-SQL benchmark converted to **DuckDB**, keeping only examples whose DuckDB query was **verified by execution**: it returns the same result as the original SQLite query on the same data.

- {train_verified} train examples (from {train_total}) over 140 databases
- {validation_verified} validation examples (from {validation_total}) over 20 different databases
- All 160 databases included as DuckDB files (`databases.zip`)

Built by the pipeline in [bhargaviHQ/huggingface-projects](https://github.com/bhargaviHQ/huggingface-projects/tree/main/duckdb-text2sql), as training data for small DuckDB text-to-SQL models.

## Why

Spider's queries are written for SQLite. SQLite accepts several things DuckDB (and standard SQL) does not, and some valid SQLite queries mean something different when run unchanged on DuckDB. Translating the SQL text is not enough; the translation has to be checked against the data.

## How it was built

For every Spider example:

1. **Copy the database.** Each SQLite database is copied into DuckDB. Column types are inferred from the stored values rather than the declared types, because SQLite does not enforce declared types. Blank strings in columns declared numeric are treated as NULL (they are missing values in Spider). Columns with any other non-numeric text become `VARCHAR`.
2. **Fix SQLite-only behaviour** in the query AST (using [sqlglot](https://github.com/tobymao/sqlglot)):
   - `double_quoted_string`: SQLite reads `"Village"` as a string literal when no column has that name. It becomes `'Village'`.
   - `bare_group_by_column`: SQLite lets you select columns that are neither grouped nor aggregated. They are added to `GROUP BY`.
   - `like_to_ilike`: SQLite `LIKE` ignores case; DuckDB `LIKE` does not. It becomes `ILIKE`.
3. **Transpile** SQLite to DuckDB with sqlglot. If the plain transpile gives a different result, the query is transpiled again with column types from the schema (this keeps SQLite integer division as integer division), first without and then with fully qualified columns.
4. **Verify by execution.** The original query runs on SQLite and the DuckDB query runs on DuckDB. An example is kept only if the results match (row order is compared only when the query has a top-level `ORDER BY`; numbers are compared to 2 decimal places). Queries with an empty result are dropped, since an empty result matches trivially.
5. **Simplify.** sqlglot adds `NULLS FIRST` to imitate SQLite's NULL ordering. It is removed again if the result does not change.
6. **Deduplicate** questions within each split.

Rejected examples are dropped, not repaired by hand. Most of them have an ambiguous answer in the original data (`ORDER BY ... LIMIT` with ties, which SQLite and DuckDB break differently) or rely on SQLite's loose typing (comparing text columns to numbers).

## Statistics

{stats_table}

## Fields

| Field | Description |
|---|---|
| `id` | `spider_{split}_{index}`, index into the original Spider split |
| `db_id` | Database name; the DuckDB file is `databases/{db_id}.duckdb` |
| `question` | Natural-language question (from Spider) |
| `sql` | Verified DuckDB query |
| `schema` | `CREATE TABLE` statements for the database, as DuckDB reports them |
| `ordered` | Whether row order is part of the answer |
| `source` | Always `spider` |
| `transpile_method` | `plain`, `typed` or `qualified` (see step 3) |
| `rewrites` | SQLite-specific rewrites applied (see step 2) |
| `original_sqlite_sql` | The original Spider query |
| `num_rows` | Number of rows the query returns |

## Usage

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import duckdb, zipfile

ds = load_dataset("bhargavis/spider-duckdb")
zipfile.ZipFile(hf_hub_download("bhargavis/spider-duckdb", "databases.zip", repo_type="dataset")).extractall(".")

ex = ds["validation"][0]
con = duckdb.connect(f"databases/{ex['db_id']}.duckdb", read_only=True)
print(ex["question"])
print(con.execute(ex["sql"]).fetchall())
```

## Limitations

- Verification proves the DuckDB query returns the same rows as the Spider gold query on this data. It does not fix errors that already exist in Spider's gold SQL, and it cannot rule out a query that coincidentally matches on this data but would differ on other data.
- Questions and schemas are Spider's. They do not exercise DuckDB-specific features such as `LIST` and `STRUCT` columns or `QUALIFY`.
- The splits follow Spider: train and validation use different databases. Spider's test set is not included.

## License and citation

Released under **CC BY-SA 4.0**, the license of Spider. If you use this dataset, please cite Spider:

```bibtex
@inproceedings{yu-etal-2018-spider,
  title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author    = {Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and Zhang, Zilin and Radev, Dragomir},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year      = {2018},
  pages     = {3911--3921}
}
```
