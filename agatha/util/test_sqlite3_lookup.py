from agatha.util.sqlite3_lookup import (
    Sqlite3LookupTable,
    Sqlite3Graph,
    Sqlite3Bow,
)
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any
import pickle

"""
The Sqlite3 Lookup table is designed to store json encoded key-value pairs.
This is designed to be as fast and flexible as necessary.
"""

def make_sqlite3_db(
    test_name:str,
    data:Dict[str, Any],
    table_name:str="lookup_table",
    key_column_name:str="key",
    value_column_name:str="value",
)->Path:
  db_path = Path("/tmp/").joinpath(f"{test_name}.sqlite3")
  if db_path.is_file():
    db_path.unlink()
  with sqlite3.connect(db_path) as db_conn:
    db_conn.execute(f"""
      CREATE TABLE {table_name} (
        {key_column_name} TEXT PRIMARY KEY,
        {value_column_name} TEXT
      );
    """)
    for key, value in data.items():
      db_conn.execute(
        f"""
          INSERT INTO {table_name}
            ({key_column_name}, {value_column_name})
            VALUES(?,?);
        """,
        (str(key), json.dumps(value))
      )
  return db_path


def test_make_sqlite3_db():
  expected = {
      "A": [1,2,3],
      "B": [4,5,6],
  }
  db_path = make_sqlite3_db("test_make_sqlite3_db", expected)
  with sqlite3.connect(db_path) as db_conn:
    actual = {}
    query_res = db_conn.execute(
        "SELECT key, value FROM lookup_table;"
    ).fetchall()
    for row in query_res:
      k, v = row
      v = json.loads(v)
      actual[k] = v
  assert actual == expected

def test_sqlite3_lookup_contains():
  expected = {
      "A": [1,2,3],
      "B": [4,5,6],
  }
  db_path = make_sqlite3_db("test_sqlite3_lookup_contains", expected)
  table = Sqlite3LookupTable(db_path)
  assert "A" in table
  assert "B" in table
  assert "C" not in table

def test_sqlite3_lookup_getitem():
  expected = {
      "A": [1,2,3],
      "B": [4,5,6],
  }
  db_path = make_sqlite3_db("test_sqlite3_lookup_getitem", expected)
  table = Sqlite3LookupTable(db_path)
  assert table["A"] == expected["A"]
  assert table["B"] == expected["B"]

def test_sqlite3_lookup_pickle():
  expected = {
      "A": [1,2,3],
      "B": [4,5,6],
  }
  db_path = make_sqlite3_db("test_sqlite3_lookup_pickle", expected)
  pickle_path = Path("/tmp/test_sqlite3_lookup_pickle.pkl")
  table = Sqlite3LookupTable(db_path)
  with open(pickle_path, 'wb') as pickle_file:
    pickle.dump(table, pickle_file)
  del table
  with open(pickle_path, 'rb') as pickle_file:
    table = pickle.load(pickle_file)
  assert table["A"] == expected["A"]
  assert table["B"] == expected["B"]
  assert "C" not in table

def test_sqlite3_lookup_preload():
  expected = {
      "A": [1,2,3],
      "B": [4,5,6],
  }
  db_path = make_sqlite3_db("test_sqlite3_lookup_preload", expected)
  table = Sqlite3LookupTable(db_path)
  # Should load the table contents to memory
  table.preload()
  # remove the in-storage database to test that content was actually loaded
  db_path.unlink()
  assert table["A"] == expected["A"]
  assert table["B"] == expected["B"]
  assert "C" not in table

def test_sqlite3_is_preloaded():
  expected = {
      "A": [1,2,3],
      "B": [4,5,6],
  }
  db_path = make_sqlite3_db("test_sqlite3_lookup_preload", expected)
  table = Sqlite3LookupTable(db_path)
  assert not table.is_preloaded()
  # Should load the table contents to memory
  table.preload()
  assert table.is_preloaded()

def test_custom_table_name():
  expected = {
      "A": [1,2,3],
      "B": [4,5,6],
  }
  db_path = make_sqlite3_db(
      "test_custom_table_name",
      expected,
      table_name="custom"
  )
  table = Sqlite3LookupTable(db_path, table_name="custom")
  assert table["A"] == expected["A"]
  assert table["B"] == expected["B"]
  assert "C" not in table


def test_custom_key_column_name():
  expected = {
      "A": [1,2,3],
      "B": [4,5,6],
  }
  db_path = make_sqlite3_db(
      "test_custom_key_column_name",
      expected,
      key_column_name="custom"
  )
  table = Sqlite3LookupTable(db_path, key_column_name="custom")
  assert table["A"] == expected["A"]
  assert table["B"] == expected["B"]
  assert "C" not in table

def test_custom_value_column_name():
  expected = {
      "A": [1,2,3],
      "B": [4,5,6],
  }
  db_path = make_sqlite3_db(
      "test_custom_value_column_name",
      expected,
      value_column_name="custom"
  )
  table = Sqlite3LookupTable(db_path, value_column_name="custom")
  assert table["A"] == expected["A"]
  assert table["B"] == expected["B"]
  assert "C" not in table

def test_backward_compatable_fallback():
  expected = {
      "A": [1,2,3],
      "B": [4,5,6],
  }
  db_path = make_sqlite3_db(
      "test_backward_compatable_fallback",
      expected,
  )
  # table set with custom (incorrect) names
  # expected behavior, fall back to defaults
  table = Sqlite3LookupTable(
      db_path,
      table_name="custom_table",
      key_column_name="custom_key",
      value_column_name="custom_value",
  )
  assert table["A"] == expected["A"]
  assert table["B"] == expected["B"]
  assert "C" not in table

def test_old_sqlite3graph():
  expected = {
      "A": ["B", "C"],
      "B": ["A"],
      "C": ["A"],
  }
  db_path = make_sqlite3_db(
      "test_old_sqlite3graph",
      expected,
      table_name="graph",
      key_column_name="node",
      value_column_name="neighbors",
  )
  table = Sqlite3Graph(db_path)
  assert table["A"] == expected["A"]
  assert table["B"] == expected["B"]
  assert table["C"] == expected["C"]
  assert "D" not in table

def test_old_sqlite3bow():
  expected = {
      "A": ["B", "C"],
      "B": ["A"],
      "C": ["A"],
  }
  db_path = make_sqlite3_db(
      "test_old_sqlite3bow",
      expected,
      table_name="sentences",
      key_column_name="id",
      value_column_name="bow",
  )
  table = Sqlite3Bow(db_path)
  assert table["A"] == expected["A"]
  assert table["B"] == expected["B"]
  assert table["C"] == expected["C"]
  assert "D" not in table

def test_new_sqlite3graph():
  expected = {
      "A": ["B", "C"],
      "B": ["A"],
      "C": ["A"],
  }
  db_path = make_sqlite3_db("test_new_sqlite3graph", expected)
  table = Sqlite3Graph(db_path)
  assert table["A"] == expected["A"]
  assert table["B"] == expected["B"]
  assert table["C"] == expected["C"]
  assert "D" not in table

def test_new_sqlite3bow():
  expected = {
      "A": ["B", "C"],
      "B": ["A"],
      "C": ["A"],
  }
  db_path = make_sqlite3_db("test_new_sqlite3bow", expected)
  table = Sqlite3Graph(db_path)
  assert table["A"] == expected["A"]
  assert table["B"] == expected["B"]
  assert table["C"] == expected["C"]
  assert "D" not in table

def test_keys():
  expected = {
      "A": ["B", "C"],
      "B": ["A"],
      "C": ["A"],
  }
  db_path = make_sqlite3_db("test_keys", expected)
  table = Sqlite3LookupTable(db_path)
  assert set(table.keys()) == set(expected.keys())

def test_len():
  expected = {
      "A": ["B", "C"],
      "B": ["A"],
      "C": ["A"],
  }
  db_path = make_sqlite3_db("test_len", expected)
  table = Sqlite3LookupTable(db_path)
  assert len(table) == len(expected)

def test_iter():
  expected = {
      "A": ["B", "C"],
      "B": ["A"],
      "C": ["A"],
  }
  db_path = make_sqlite3_db("test_iter", expected)
  table = Sqlite3LookupTable(db_path)
  actual = {k: v for k, v in table}
  assert actual == expected

def test_iter_where():
  db_data = {
      "AA": ["B", "C"],
      "BBBB": ["A"],
      "CC": ["A"],
  }
  db_path = make_sqlite3_db("test_iter_where", db_data)
  table = Sqlite3LookupTable(db_path)
  actual = {k: v for k, v in table.iterate(where="length(key) = 2")}
  expected= {
      "AA": ["B", "C"],
      "CC": ["A"],
  }
  assert actual == expected
