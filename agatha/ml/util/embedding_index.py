from typing import List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
import h5py
from tqdm import tqdm
import sqlite3
from typing import Iterable, Tuple
from itertools import chain

@dataclass
class EmbeddingLocation:
  "This class stores information needed to load an embedding"
  entity_type: str
  partition_idx: int
  row_idx: Optional[int] = None
  def __iter__(self):
    yield self.entity_type
    yield self.partition_idx
    if self.row_idx is not None:
      yield self.row_idx


class EmbeddingLocationIndex(object):
  """
  This class manages pulling embedding locations from the sqlite3 database
  """
  def __init__(self,
      db_path:Path,
      db_name:str="embedding_locations",
      preload=False
  ):
    self.preload=preload
    db_path = Path(db_path)
    assert db_path.is_file(), "Invalid path to database."
    self.db_path = db_path
    self.db_name = db_name
    self.db_conn = None
    self.db_cursor = None
    self.select_fmt_str = """
      SELECT
        entity_type,
        partition_idx,
        row_idx
      FROM
        {db_name}
      WHERE
        entity=?
      ;
    """
    self._cache = {}

  def _get_or_none(self, entity:str):
    return self.db_cursor.execute(
        self.select_fmt_str.format(db_name=self.db_name),
        (entity,)
    ).fetchone()

  def __contains__(self, entity:str)->bool:
    assert self.db_cursor is not None, "__contains__ called outside of with"
    res = self._cache.get(entity, "Missing")
    if res == "Missing":
      res = self._cache[entity] = self._get_or_none(entity)
    return res is not None

  def __getitem__(self, entity:str)->EmbeddingLocation:
    assert self.db_cursor is not None, "__getitem__ called outside of with"
    assert entity in self
    return self._cache[entity]

  def _query_to_emb_loc(self, cursor, row)->EmbeddingLocation:
      return EmbeddingLocation(*row)

  def __enter__(self):
    self.db_conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
    self.db_conn.execute('PRAGMA journal_mode = OFF')
    self.db_conn.execute('PRAGMA synchronous = OFF')
    self.db_conn.execute('PRAGMA cache_size = 100000')
    self.db_conn.execute('PRAGMA temp_store = MEMORY')

    if self.preload:
      self._cache = {
          entity: EmbeddingLocation(
                    entity_type=entity_type,
                    partition_idx=partition_idx,
                    row_idx=row_idx
                  )
          for entity, entity_type, partition_idx, row_idx
          in self.db_conn.execute(f"""
            SELECT entity, entity_type, partition_idx, row_idx
            FROM {self.db_name};
          """).fetchall()
      }

    self.db_conn.row_factory = self._query_to_emb_loc
    self.db_cursor = self.db_conn.cursor()
    return self

  def get_names_of_type(self, typ_char:str)->List[str]:
    res = []
    for ent, val in self._cache.items():
      if val is not None and ent[0] == typ_char:
        res.append(ent)
    return res

  def __exit__(self, exc_type, exc_value, traceback):
    self.db_conn.close()
    self.db_conn = None
    self.db_cursor = None
    return False

class EmbeddingIndex(object):
  def __init__(
      self,
      embedding_dir:Path,
      embedding_location_db_path:Path,
      emb_ver:str=None
  ):
    embedding_dir = Path(embedding_dir)
    embedding_location_db_path = Path(embedding_location_db_path)
    # Setup entity->location index index
    self.embedding_location_index = EmbeddingLocationIndex(
        embedding_location_db_path,
    )
    self.inside_context_mngr = False

    # This dir holds embedding files
    self.embedding_dir = Path(embedding_dir)
    assert self.embedding_dir.is_dir(), "Invalid embedding_dir"

    # Setup embedding version
    valid_emb_ver = self.load_embedding_versions(embedding_dir)
    assert len(valid_emb_ver) > 0, "Invalid embedding dir, has no versions."
    if emb_ver is None:
      assert len(valid_emb_ver) == 1, \
        f"Must supply emb_ver if multiple exist: {valid_emb_ver}"
      self.emb_ver = next(iter(valid_emb_ver))
    else:
      assert emb_ver in valid_emb_ver, "Invalid emb_ver"
      self.emb_ver = emb_ver

    self._cache =  {}

  @staticmethod
  def load_embedding_versions(embedding_dir:Path)->Set[str]:
    return set(map(
      lambda p: p.name.split(".")[1],
      embedding_dir.glob("embeddings_*.v*.h5")
    ))

  def __enter__(self):
    self.inside_context_mngr = True
    self.embedding_location_index.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.inside_context_mngr = False
    self.embedding_location_index.__exit__(exc_type, exc_value, traceback)
    return False # Don't want to handle exceptions

  def __contains__(self, name:str)->bool:
    assert self.inside_context_mngr, "Called __contains__ outside of with"
    return name in self.embedding_location_index

  def __getitem__(self, name:str)->np.array:
    assert self.inside_context_mngr, "Called __getitem__ outside of with"
    res = self._cache.get(name)
    if res is None:
      emb_loc = self.embedding_location_index[name]
      assert emb_loc is not None, f"EmbeddingIndex does not contain {name}"
      res = self._cache[name] = self._load_embedding_from_h5(emb_loc)
    return res

  def _load_embedding_from_h5(self, emb_loc:EmbeddingLocation)->np.array:
    h5_path = self._get_embedding_path(emb_loc)
    assert h5_path.is_file(), f"{emb_loc} -> {h5_path} Does not exist."
    with h5py.File(h5_path, "r") as h5_file:
      return h5_file["embeddings"][emb_loc.row_idx]

  def _get_embedding_path(self, el:EmbeddingLocation)->Path:
    return self.embedding_dir.joinpath(
        f"embeddings_{el.entity_type}_{el.partition_idx}.{self.emb_ver}.h5"
    )


class PreloadedEmbeddingIndex(object):
  def __init__(
      self,
      embedding_dir:Path,
      embedding_location_db_path:Path,
      entity_types:str,  # one char per ent
      emb_ver:str=None,
  ):
    embedding_dir=Path(embedding_dir)
    self.entity_index = EmbeddingLocationIndex(
        embedding_location_db_path,
        preload=True,
    ).__enter__()

    # Setup embedding version
    valid_emb_ver = EmbeddingIndex.load_embedding_versions(embedding_dir)
    assert len(valid_emb_ver) > 0, "Invalid embedding dir, has no versions."
    if emb_ver is None:
      assert len(valid_emb_ver) == 1, \
        f"Must supply emb_ver if multiple exist: {valid_emb_ver}"
      self.emb_ver = next(iter(valid_emb_ver))
    else:
      assert emb_ver in valid_emb_ver, "Invalid emb_ver"
      self.emb_ver = emb_ver

    self.embedding_paths = list(chain.from_iterable([
      embedding_dir.glob(f"embeddings_{typ}_*.{self.emb_ver}.h5")
      for typ in entity_types
    ]))
    assert len(self.embedding_paths) > 0
    print("Loading Embeddings")
    self.loc2embeddings = {}
    for path in tqdm(self.embedding_paths):
      loc = self.parse_embedding_path(path)
      with h5py.File(path, "r") as h5_file:
        self.loc2embeddings[tuple(loc)] = h5_file["embeddings"][()]

  def __getitem__(self, name:str)->np.array:
    loc = self.entity_index[name]
    return self.loc2embeddings[
        (loc.entity_type, loc.partition_idx)
    ][loc.row_idx]

  def __len__(self)->int:
    return len(self.entity_index)

  def __contains__(self, name:str)->bool:
    return name in self.entity_index

  @staticmethod
  def parse_embedding_path(embedding_path:str)->EmbeddingLocation:
    entity_type, part_idx = embedding_path.name.split(".",1)[0].split("_")[1:]
    return EmbeddingLocation(
        entity_type=entity_type,
        partition_idx=int(part_idx),
    )
