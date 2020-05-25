from pathlib import Path
from torch.utils.data import Dataset
from agatha.util.sqlite3_lookup import Sqlite3LookupTable
from typing import Tuple, Any, Callable, List, Union

def _accept_all(key:str)->bool:
  return True

class Sqlite3Dataset(Dataset):
  def __init__(
      self,
      table:Union[Sqlite3LookupTable, Path],
      filter_fn:Callable[[str], bool]=_accept_all,
  ):
    if isinstance(table, str) or isinstance(table, Path):
      table = Sqlite3LookupTable(table)

    self.table=table
    self._keys = None
    self._filter_fn = filter_fn
    self._len = None

  def __getstate__(self):
    "Don't serialize the keys."
    keys = self._keys
    self._keys = None
    state = self.__dict__.copy()
    self._keys = keys
    return state

  def keys(self)->List[str]:
    if self._keys is None:
      self._keys = [
          k for k in self.table.keys()
          if self._filter_fn(k)
      ]
    return self._keys

  def __len__(self):
    return len(self.keys())

  def __getitem__(self, idx:int)->Tuple[str, Any]:
    assert 0 <= idx < len(self), "Invalid index."
    key = self.keys()[idx]
    value = self.table[key]
    return key, value

class Sqlite3KeyDataset(Sqlite3Dataset):
  """
  Same as above, only returns keys
  """
  def __init__(self, *args, **kwargs):
    Sqlite3Dataset.__init__(self, *args, **kwargs)

  def __getitem__(self, idx)->str:
    return Sqlite3Dataset.__getitem__(self, idx)[0]

class Sqlite3ValueDataset(Sqlite3Dataset):
  """
  Same as above, only returns values
  """
  def __init__(self, *args, **kwargs):
    Sqlite3Dataset.__init__(self, *args, **kwargs)

  def __getitem__(self, idx)->str:
    return Sqlite3Dataset.__getitem__(self, idx)[1]
