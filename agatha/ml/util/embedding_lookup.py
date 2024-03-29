from agatha.util.sqlite3_lookup import Sqlite3LookupTable
from pathlib import Path
from typing import Tuple, Set
import h5py
import numpy as np

def parse_embedding_path(path:Path)->Tuple[str, int]:
  """
  Given a path to an embedding hdf5 file with a name like:
  embeddings_s_99.v5.h5
  return (entity_type, partition_index)
  """
  stem, ver, suffix = path.name.split(".")
  assert suffix == "h5", "Invalid file type"
  assert ver[0] == "v", "Invalid file name, 2nd component must be version"

  embeddings, typ, part = stem.split("_")
  assert embeddings == "embeddings", "Invalid file name"
  assert len(typ) == 1, "Invalid file name"
  part = int(part)
  return typ, part

class EmbeddingLookupTable():
  def __init__(
      self,
      embedding_dir:Path,
      entity_db:Path,
      disable_cache:bool=False,
  ):
    embedding_dir = Path(embedding_dir)
    entity_db = Path(entity_db)
    assert embedding_dir.is_dir(), "Failed to find embedding_dir"
    assert entity_db.is_file(), "Failed to find entities"
    self.entities = Sqlite3LookupTable(entity_db, disable_cache=disable_cache)
    self._type_part2path = {
        parse_embedding_path(embedding_path): embedding_path
        for embedding_path
        in embedding_dir.glob("embeddings_*.h5")
    }
    assert any(self._type_part2path), "Failed to find embedding files."
    self._type_part2file_handle = {}
    self._type_part2matrix = {}
    self._use_cache = not disable_cache

  def __del__(self)->None:
    self._close_file_handles()

  def _close_file_handles(self)->None:
    for handle in self._type_part2file_handle.values():
      handle.close()
    self._type_part2file_handle.clear()

  def __getstate__(self):
    "If we pickle, don't pickle preloaded data"
    self._close_file_handles()
    cached_data = self._type_part2matrix
    self._type_part2matrix = {}
    state = self.__dict__.copy()
    self._type_part2matrix = cached_data
    return state

  def clear_cache(self)->None:
    self._type_part2matrix.clear()

  def disable_cache(self)->None:
    self.clear_cache()
    self._use_cache = False

  def enable_cache(self)->None:
    self._use_cache = True

  def _cache_matrix(self, type_:str, part:int)->None:
    path_key = (type_, part)
    if self._use_cache and path_key not in self._type_part2matrix:
      self._type_part2matrix[path_key] = \
          self._get_file_handle(*path_key)["embeddings"][()]

  def _get_file_handle(self, type_:str, part:int)->h5py.File:
    path_key = (type_, part)
    assert path_key in self._type_part2path, \
      f"Cannot find path associated with: {entity} --- {location}"
    if path_key not in self._type_part2file_handle:
      h5_path = self._type_part2path[path_key]
      assert h5_path.is_file(),  f"Missing file: {h5_path}"
      self._type_part2file_handle[path_key] = h5py.File(h5_path, "r")
    return self._type_part2file_handle[path_key]

  def _get_row(self, type_:str, part:int, row:int)->np.array:
    # Adds path_key to _type_part2matrix if using cache
    path_key = (type_, part)
    self._cache_matrix(*path_key)
    if path_key in self._type_part2matrix:
      return self._type_part2matrix[path_key][row]
    else:  # user has elected not to use cache
      return self._get_file_handle(*path_key)["embeddings"][row]
  
  def _get_row_nocache(self, type_:str, part:int, row:int)->np.array:
    # Doesn't use any caching (we already got everything)
    path_key = (type_, part)
    return self._get_file_handle(*path_key)["embeddings"][row]

  def __getitem__(self, entity:str)->np.array:
    assert entity in self.entities, f"Cannot find {entity} in index"
    location = self.entities[entity]
    assert "type" in location, f"Invalid location object: {location}"
    assert "part" in location, f"Invalid location object: {location}"
    assert "row" in location, f"Invalid location object: {location}"
    type_ = str(location["type"])
    part = int(location["part"])
    row = int(location["row"])
    return self._get_row(type_, part, row)

  def __contains__(self, entity:str)->bool:
    return entity in self.entities

  def preload(self)->None:
    if not self.is_preloaded():
      self.entities.preload()
      for path_key in self._type_part2path:
        self._cache_matrix(*path_key)

  def is_preloaded(self)->bool:
    "the entity index is loaded and all paths have been loaded"
    return (
        self.entities.is_preloaded()
        and (
          set(self._type_part2matrix.keys())
          == set(self._type_part2path.keys())
        )
    )

  def keys(self)->Set[str]:
    return self.entities.keys()
