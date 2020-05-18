from itertools import zip_longest
from typing import List, Any, Dict
import numpy as np
import hashlib

Record = Dict[str, Any]


def iter_to_batches(iterable, batch_size):
  """
  Chunks the input iterable into fixed-sized batches.
  Example:
    ```python3
    list(iter_to_batches(range(10), 3))
    [
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
      [9]
    ]
    ```
  """
  args = [iter(iterable)] * batch_size
  for batch in zip_longest(*args):
    yield list(filter(lambda b: b is not None, batch))


def generator_to_list(*args, gen_fn=None, **kwargs):
  return [
      val for val in gen_fn(*args, **kwargs)
  ]


def flatten_list(list_of_lists:List[List[Any]])->List[Any]:
  return [item for sublist in list_of_lists for item in sublist]


def hash_str_to_int64(s):
  return np.int64(
      int.from_bytes(
        hashlib.md5(
          s.encode("utf-8")
        ).digest(),
        byteorder="big",
        signed=False,
      ) % np.iinfo(np.int64).max
  )


def hash_str_to_int32(s):
  return np.int32(
      int.from_bytes(
        hashlib.md5(
          s.encode("utf-8")
        ).digest(),
        byteorder="big",
        signed=False,
      ) % np.iinfo(np.int32).max
  )

def hash_str_to_int(s):
  return int.from_bytes(
      hashlib.md5(s.encode("utf-8")).digest(),
      byteorder="big",
      signed=False,
  ) % np.iinfo(np.int64).max


def merge_counts(
  key_to_doc_count_1:Record,
  key_to_doc_count_2:Record=None,
)->Record:
  "Adds up counts from two dicts"
  if key_to_doc_count_2 is None:
    return key_to_doc_count_1
  if len(key_to_doc_count_1) > len(key_to_doc_count_2):
    larger, smaller = key_to_doc_count_1, key_to_doc_count_2
  else:
    larger, smaller = key_to_doc_count_2, key_to_doc_count_1
  for key, count in smaller.items():
    if key in larger:
      larger[key] += count
    else:
      larger[key] = count
  return larger
