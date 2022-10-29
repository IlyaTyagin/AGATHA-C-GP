from agatha.ml.util.embedding_lookup import EmbeddingLookupTable
from agatha.util.sqlite3_lookup import Sqlite3LookupTable
import numpy as np
from typing import List, Tuple, Set, Dict, Any
from agatha.util.entity_types import (
    PREDICATE_TYPE,
    UMLS_TERM_TYPE,
    is_predicate_type,
    is_umls_term_type,
)
import random
import torch
from dataclasses import dataclass
import time

from pathlib import Path
import copy
import json
from tqdm import tqdm

@dataclass
class PredicateEmbeddings:
  subj:np.array
  obj:np.array
  subj_neigh:List[np.array]
  obj_neigh:List[np.array]


def clean_coded_term(term:str)->str:
  """
  If term is not formatted as an agatha coded term key, produces a coded term
  key. Otherwise, just returns the term.
  """
  if is_umls_term_type(term):
    return term.lower()
  else:
    return f"{UMLS_TERM_TYPE}:{term}".lower()


def is_valid_predicate_name(predicate_name:str)->bool:
  if not is_predicate_type(predicate_name):
    return False
  try:
    typ, sub, vrb, obj = predicate_name.lower().split(":")
  except Exception:
    return False
  return (len(sub) > 0) and (len(obj) > 0)


def parse_predicate_name(predicate_name:str)->Tuple[str, str]:
  """Parses subject and object from predicate name strings.

  Predicate names are formatted strings that follow this convention:
  p:{subj}:{verb}:{obj}. This function extracts the subject and object and
  returns coded-term names in the form: m:{entity}. Will raise an exception if
  the predicate name is improperly formatted.

  Args:
    predicate_name: Predicate name in form p:{subj}:{verb}:{obj}.

  Returns:
    The subject and object formulated as coded-term names.

  """
  assert is_predicate_type(predicate_name), \
      f"Not a predicate name: {predicate_name}"
  typ, sub, vrb, obj = predicate_name.lower().split(":")
  assert typ == PREDICATE_TYPE
  return clean_coded_term(sub), clean_coded_term(obj)


def to_predicate_name(
    subj:str,
    obj:str,
    verb:str="unknown",
    )-> str:
  """Converts two names into a predicate of form p:t1:verb:t2

  Assumes that terms are correct Agatha graph keys. This means that we expect
  input terms in the form of m:____. Allows for a custom verb type, but
  defaults to unknown. Output will always be set to lowercase.

  Example usage:

  ```
  to_predicate_name(m:c1, m:c2)
  > p:c1:unknown:c2
  to_predicate_name(m:c1, m:c2, "treats")
  > p:c1:treats:c2
  to_predicate_name(m:c1, m:c2, "TREATS")
  > p:c1:treats:c2
  ```

  Args:
    subj: Subject term. In the form of "m:_____"
    obj: Object term. In the form of "m:_____"
    verb: Optional verb term for resulting predicate.

  Returns:
    Properly formatted predicate containing subject and object. Verb type will
    be set to "UNKNOWN"

  """
  assert is_umls_term_type(subj), \
    f"Called to_predicate_name with bad subject: {subj})"
  assert is_umls_term_type(obj), \
    f"Called to_predicate_name with bad object: {obj})"
  assert ":" not in verb, "Verb cannot contain colon character"
  subj = subj[2:]
  obj = obj[2:]
  return f"{PREDICATE_TYPE}:{subj}:{verb}:{obj}".lower()



class PredicateObservationGenerator():
  """
  Converts predicate names to predicate observations
  """
  def __init__(
      self,
      graph:Sqlite3LookupTable,
      embeddings:EmbeddingLookupTable,
      neighbor_sample_rate:int,
  ):
    assert neighbor_sample_rate >= 0
    self.graph = graph
    self.embeddings = embeddings
    self.neighbor_sample_rate = neighbor_sample_rate

  def _sample_neighborhood(self, neigh:Set[str])->List[str]:
    neigh = list(neigh)
    if len(neigh) < self.neighbor_sample_rate:
      return neigh
    else:
      return random.sample(neigh, self.neighbor_sample_rate)

  def _get_pred_neigh_from_diff(
      self,
      subj:str,
      obj:str
  )->Tuple[List[str], List[str]]:
    assert subj in self.graph, f"Failed to find {subj} in graph."
    assert obj in self.graph, f"Failed to find {obj} in graph."
    s = set(filter(is_predicate_type, self.graph[subj]))
    o = set(filter(is_predicate_type, self.graph[obj]))
    s, o = (s-o, o-s)
    return self._sample_neighborhood(s), self._sample_neighborhood(o)

  def __getitem__(self, predicate:str)->PredicateEmbeddings:
    try:
      subj, obj = parse_predicate_name(predicate)
    except Exception:
      raise Exception(f"Failed to parse predicate: {predicate}")
    start = time.time()
    subj_neigh, obj_neigh = self._get_pred_neigh_from_diff(subj, obj)
    subj = self.embeddings[subj]
    obj = self.embeddings[obj]
    subj_neigh = [self.embeddings[s] for s in subj_neigh]
    obj_neigh = [self.embeddings[o] for o in obj_neigh]
    end = time.time()
    #print("Generating PredicateEmbeddings:", int(end-start))
    return PredicateEmbeddings(
        subj=subj,
        obj=obj,
        subj_neigh=subj_neigh,
        obj_neigh=obj_neigh
    )


class PredicateScrambleObservationGenerator(PredicateObservationGenerator):
  """
  Same as above, but the neighborhood comes from randomly selected predicates
  """
  def __init__(self, predicates:List[str], *args, **kwargs):
    PredicateObservationGenerator.__init__(self, *args, **kwargs)
    self.predicates = predicates

  def __getitem__(self, predicate:str):
    subj, obj = parse_predicate_name(predicate)
    subj = self.embeddings[subj]
    obj = self.embeddings[obj]
    neighs = [
        self.embeddings[predicate]
        for predicate in
        random.sample(self.predicates, self.neighbor_sample_rate*2)
    ]
    return PredicateEmbeddings(
        subj=subj,
        obj=obj,
        subj_neigh=neighs[:self.neighbor_sample_rate],
        obj_neigh=neighs[self.neighbor_sample_rate:]
    )


class NegativePredicateGenerator():
  def __init__(
      self,
      coded_terms:List[str],
      graph:Sqlite3LookupTable,
  ):
    "Generates coded terms that appear in graph."
    self.coded_terms = coded_terms
    self.graph = graph

  def _choose_term(self):
    term = random.choice(self.coded_terms)
    while term not in self.graph:
      term = random.choice(self.coded_terms)
    return term

  def generate(self):
    subj = self._choose_term()
    obj = self._choose_term()
    predicate = to_predicate_name(subj, obj)
    return predicate


class PredicateExampleDataset(torch.utils.data.Dataset):
  def __init__(
      self,
      predicate_ds:torch.utils.data.Dataset,
      all_predicates:List[str],
      graph:Sqlite3LookupTable,
      embeddings:EmbeddingLookupTable,
      coded_terms:List[str],
      neighbor_sample_rate:int,
      negative_swap_rate:int,
      negative_scramble_rate:int,
      preload_on_first_call:bool=True,
      verbose:bool=False,
  ):
    self.graph = graph
    self.embeddings = embeddings
    self.verbose = verbose
    self.predicate_ds = predicate_ds
    self.negative_generator = NegativePredicateGenerator(
        coded_terms=coded_terms,
        graph=graph,
    )
    self.scramble_observation_generator = PredicateScrambleObservationGenerator(
        predicates=all_predicates,
        graph=graph,
        embeddings=embeddings,
        neighbor_sample_rate=neighbor_sample_rate,
    )
    self.observation_generator = PredicateObservationGenerator(
        graph=graph,
        embeddings=embeddings,
        neighbor_sample_rate=neighbor_sample_rate,
    )
    self.negative_swap_rate = negative_swap_rate
    self.negative_scramble_rate  = negative_scramble_rate
    self._first_call = preload_on_first_call

  def __len__(self)->int:
    return len(self.predicate_ds)

  def __getitem__(self, idx:int)->Dict[str, Any]:
    if self._first_call:
      print("Worker preloading...")
      start = time.time()
      self.graph.preload()
      self.embeddings.preload()
      end = time.time()
      print(f"Worker preloading: {int(end-start)}s")
      self._first_call = False
    start = time.time()
    positive_predicate = self.predicate_ds[idx]
    positive_observation = self.observation_generator[positive_predicate]
    negative_predicates = []
    negative_observations = []
    for _ in range(self.negative_swap_rate):
      p = self.negative_generator.generate()
      negative_predicates.append(p)
      negative_observations.append(self.observation_generator[p])
    for _ in range(self.negative_swap_rate):
      p = self.negative_generator.generate()
      negative_predicates.append(p)
      negative_observations.append(self.scramble_observation_generator[p])
    end = time.time()
    #print(f"Worker produced batch: {int(end-start)}")
    return dict(
        positive_predicate=positive_predicate,
        positive_observation=positive_observation,
        negative_predicates=negative_predicates,
        negative_observations=negative_observations,
    )



def collate_predicate_embeddings(
    predicate_embeddings:List[PredicateEmbeddings]
):
  return torch.cat([
    torch.nn.utils.rnn.pad_sequence([
      torch.FloatTensor([p.subj, p.obj] + p.subj_neigh + p.obj_neigh)
      for p in predicate_embeddings
    ])
  ])

def collate_predicate_training_examples(
    examples:List[Dict[str,Any]],
)->Dict[str, Any]:
  """
  Takes a list of results from PredicateExampleDataset and produces tensors
  for input into the agatha training model.
  """
  positive_predicates = [e["positive_predicate"] for e in examples]
  positive_observations = collate_predicate_embeddings(
      [e["positive_observation"] for e in examples]
  )
  negative_predicates_list = \
      list(zip(*[e["negative_predicates"] for e in examples]))
  negative_observations_list = [
      collate_predicate_embeddings(neg_obs)
      for neg_obs in zip(*[e["negative_observations"] for e in examples])
  ]
  return dict(
      positive_predicates=positive_predicates,
      positive_observations=positive_observations,
      negative_predicates_list=negative_predicates_list,
      negative_observations_list=negative_observations_list,
  )

class Numpy_cache_links_obj():
    
    def __init__(self, embeddings):
        
        self.json_dataloader_list = []
        
        self.n_samples = 0
        self.neigh_sample_rate = 0
        self.neg_per_batch = 0
        
        self.current_dtype = np.uint32
        
        self.type_to_n_dict = {
            'p': 1,
            'm': 2,
            'e': 3,
            'n': 4,
            'l': 5,
            's': 6,
        }

        self.n_to_type_dict = {
            v:k for k,v in self.type_to_n_dict.items()
        }
        
        self.pos_subj_np = None
        self.dataloader_np_dict = None
        
        self.positive_predicate_list = []
        self.negative_predicates_list = []
        
        self.json_dataloader_reduced_list = []
        
        self.emb_lookup_table = embeddings
        
        return None
    
    def load_json_ckpts(self, fnames_list):
        json_dataloader_list = []
        for fname in tqdm(fnames_list):
            with open(fname, 'r') as f:
                for line in f:
                    json_dataloader_list.append(
                        json.loads(line)
                    )
                    
        self.json_dataloader_list = json_dataloader_list
        self.json_dataloader_reduced_list = []

        for k in json_dataloader_list:
            self.json_dataloader_reduced_list.append(k['value'])
            self.positive_predicate_list.append(
                k['value']['positive_predicate']
            )
            self.negative_predicates_list.append(
                k['value']['negative_predicates']
            )
        return None
    
    def get_size_params(self):
        
        if self.json_dataloader_reduced_list:
            self.n_samples = len(self.json_dataloader_reduced_list)

            self.neigh_sample_rate = max(
                [len(l['positive_observation']['subj_neigh']) for l in self.json_dataloader_reduced_list]
            )

            self.neg_per_batch = max(
                [len(l['negative_observations']) for l in self.json_dataloader_reduced_list]
            )
        elif self.neg_subj_neigh_np.shape:
            (self.n_samples, 
            self.neg_per_batch, 
            self.neigh_sample_rate, _ )= (
                self.neg_subj_neigh_np.shape
            )
        else:
            print('Object it empty!')
        return None
    
    def init_empty_numpy_arrays(
        self, 
        n_samples,
        neg_per_batch, 
        neigh_sample_rate, 
        current_dtype
    ):
        
        self.pos_subj_np = np.zeros(shape=(n_samples, 3), dtype=current_dtype)
        self.pos_obj_np = np.zeros(shape=(n_samples, 3), dtype=current_dtype)
        self.pos_subj_neigh_np = np.zeros(shape=(n_samples, neigh_sample_rate, 3), dtype=current_dtype)
        self.pos_obj_neigh_np = np.zeros(shape=(n_samples, neigh_sample_rate, 3), dtype=current_dtype)

        self.neg_subj_np = np.zeros(shape=(n_samples, neg_per_batch, 3), dtype=current_dtype)
        self.neg_obj_np = np.zeros(shape=(n_samples, neg_per_batch, 3), dtype=current_dtype)
        self.neg_subj_neigh_np = np.zeros(shape=(n_samples, neg_per_batch, neigh_sample_rate, 3), dtype=current_dtype)
        self.neg_obj_neigh_np = np.zeros(shape=(n_samples, neg_per_batch, neigh_sample_rate, 3), dtype=current_dtype)
        
        return None
    
    #----lowest lvl----#
    def prt_dict_to_np(self, prt_dict):
    
        return [
            prt_dict['part'],
            prt_dict['row'],
            self.type_to_n_dict[prt_dict['type']]
        ]

    def np_to_prt_dict(self, np_arr):

        return {
            'part': np_arr[0],
            'row': np_arr[1],
            'type': self.n_to_type_dict[np_arr[2]]
        }
    #--------#
    
    def decode_json_dl_list(self, json_dataloader_list):
        for idx, el in enumerate(tqdm(json_dataloader_list)):
            el_dict = el['value']

            # positive
            self.pos_subj_np[idx] = self.prt_dict_to_np(el_dict['positive_observation']['subj'])
            self.pos_obj_np[idx]  = self.prt_dict_to_np(el_dict['positive_observation']['obj'] )

            for neigh_idx, pred in enumerate(el_dict['positive_observation']['subj_neigh']):
                self.pos_subj_neigh_np[idx, neigh_idx] = self.prt_dict_to_np(pred)

            for neigh_idx, pred in enumerate(el_dict['positive_observation']['obj_neigh']):
                self.pos_obj_neigh_np[idx, neigh_idx] = self.prt_dict_to_np(pred)


            # negative
            for neg_sample_idx, neg_sample in enumerate(el_dict['negative_observations']):
                self.neg_subj_np[idx, neg_sample_idx] = self.prt_dict_to_np(neg_sample['subj'])
                self.neg_obj_np[idx, neg_sample_idx] = self.prt_dict_to_np(neg_sample['obj'])

                for neigh_idx, pred in enumerate(neg_sample['subj_neigh']):
                    self.neg_subj_neigh_np[idx, neg_sample_idx, neigh_idx] = self.prt_dict_to_np(pred)

                for neigh_idx, pred in enumerate(neg_sample['obj_neigh']):
                    self.neg_obj_neigh_np[idx, neg_sample_idx, neigh_idx] = self.prt_dict_to_np(pred)
    
        return None
    
    def Construct_dataloader_np_dict(self):
    
        self.dataloader_np_dict = {
            'positive_observation': {
                'subj': self.pos_subj_np,
                'obj': self.pos_obj_np,
                'subj_neigh': self.pos_subj_neigh_np,
                'obj_neigh': self.pos_obj_neigh_np,
            },
            'negative_observations':{
                'subj': self.neg_subj_np,
                'obj': self.neg_obj_np,
                'subj_neigh': self.neg_subj_neigh_np,
                'obj_neigh': self.neg_obj_neigh_np,
            }
        }

        return None
    
    def from_jsons(self, fnames_list):
        self.load_json_ckpts(fnames_list)
        self.get_size_params()
        self.init_empty_numpy_arrays(
            self.n_samples,
            self.neg_per_batch, 
            self.neigh_sample_rate, 
            self.current_dtype
        )
        self.decode_json_dl_list(
            self.json_dataloader_list
        )
        self.Construct_dataloader_np_dict()
        
        return None
    
    def np_to_sample(self, idx):
        return_dict = dict()
        
        return_dict['positive_predicate'] = (
            self.positive_predicate_list[idx]
        )
        
        return_dict['positive_observation'] = {
            'subj': self.np_to_prt_dict(self.dataloader_np_dict['positive_observation']['subj'][idx]),
            'obj':  self.np_to_prt_dict(self.dataloader_np_dict['positive_observation']['obj'][idx]),
            'subj_neigh': [
                self.np_to_prt_dict(pos_neigh) for pos_neigh in self.dataloader_np_dict['positive_observation']['subj_neigh'][idx] if pos_neigh[-1]
            ],
            'obj_neigh': [
                self.np_to_prt_dict(pos_neigh) for pos_neigh in self.dataloader_np_dict['positive_observation']['obj_neigh'][idx] if pos_neigh[-1]
            ],        
        }

        neg_subj = [self.np_to_prt_dict(neg_node) for neg_node in self.dataloader_np_dict['negative_observations']['subj'][idx]]
        neg_obj = [self.np_to_prt_dict(neg_node) for neg_node in self.dataloader_np_dict['negative_observations']['obj'][idx]]

        neg_subj_neigh = []
        neg_obj_neigh = []
        for neg_idx, _ in enumerate(self.dataloader_np_dict['negative_observations']['subj_neigh'][idx]):
            cur_subj_neigh = []
            cur_obj_neigh = []
            for sample_idx, _ in enumerate(self.dataloader_np_dict['negative_observations']['subj_neigh'][idx, neg_idx]):
                cur_subj_neigh_np = self.dataloader_np_dict['negative_observations']['subj_neigh'][idx, neg_idx, sample_idx]
                cur_obj_neigh_np = self.dataloader_np_dict['negative_observations']['obj_neigh'][idx, neg_idx, sample_idx]
                if cur_subj_neigh_np[-1]:
                    cur_subj_neigh.append(self.np_to_prt_dict(cur_subj_neigh_np))
                if cur_obj_neigh_np[-1]:
                    cur_obj_neigh.append(self.np_to_prt_dict(cur_obj_neigh_np))
            neg_subj_neigh.append(cur_subj_neigh)
            neg_obj_neigh.append(cur_obj_neigh)
            
        return_dict['negative_predicates'] = (
            self.negative_predicates_list[idx]
        )

        return_dict['negative_observations'] = [
            {
                'subj': neg_subj[neg_idx],
                'obj': neg_obj[neg_idx],
                'subj_neigh': neg_subj_neigh[neg_idx],
                'obj_neigh': neg_obj_neigh[neg_idx],
            } for neg_idx, _ in enumerate(neg_subj)
        ]

        return return_dict
    
    def Dict_to_PredEmb(self, sample_dict)->PredicateEmbeddings:
        return PredicateEmbeddings(
            subj = sample_dict['subj'],
            obj = sample_dict['obj'],
            subj_neigh = sample_dict['subj_neigh'],
            obj_neigh = sample_dict['obj_neigh']
        )
    
    def convert_h5_link_to_vect(self, obs_orig):

      obs = copy.copy(obs_orig)

      def get_emb(emb_loc_dict):

          node_type = emb_loc_dict['type']
          node_part = emb_loc_dict['part']
          node_row = emb_loc_dict['row']

          vect = self.emb_lookup_table._get_row_nocache(
              node_type,
              node_part,
              node_row
          )

          return vect

      obs.subj = get_emb(obs.subj)
      obs.obj = get_emb(obs.obj)
      obs.subj_neigh = [get_emb(l) for l in obs.subj_neigh]
      obs.obj_neigh = [get_emb(l) for l in obs.obj_neigh]

      return obs

    def emb_one_sample(
      self, 
      cache_cpkt:dict
    )->Dict[str, Any]:

      train_ckpt_decoded = copy.deepcopy(cache_cpkt)
      train_ckpt_decoded['positive_predicate'] = cache_cpkt['positive_predicate']

      train_ckpt_decoded['positive_observation'] = (
          self.convert_h5_link_to_vect(
            cache_cpkt['positive_observation']
          )
      )

      train_ckpt_decoded['negative_predicates'] = cache_cpkt['negative_predicates']

      nos_list = []

      for no in train_ckpt_decoded['negative_observations']:
          nos_list.append(
              self.convert_h5_link_to_vect(no)
          )

      train_ckpt_decoded['negative_observations'] = nos_list

      return train_ckpt_decoded
    
    def get_PredEmb_item(self, single_sample)->dict:

        single_sample['positive_observation'] = self.Dict_to_PredEmb(
          single_sample['positive_observation']
        )
        single_sample['negative_observations'] = (
          [
            self.Dict_to_PredEmb(p) for p in single_sample['negative_observations']
          ]
        )
        return single_sample
    
    def dump(self, fpath):
        
        fpath = Path(fpath)
        fpath.mkdir(parents=True, exist_ok=True)
        
        with open(fpath.joinpath('lists.json'), 'w') as f:
            json.dump(
                {
                    'positive_predicate_list': self.positive_predicate_list,
                    'negative_predicates_list': self.negative_predicates_list,
                },
                f
            )
        np.savez(
            fpath.joinpath('arrays.npz'),
            pos_subj_np = self.pos_subj_np,
            pos_obj_np = self.pos_obj_np,
            pos_subj_neigh_np = self.pos_subj_neigh_np,
            pos_obj_neigh_np = self.pos_obj_neigh_np,
            neg_subj_np = self.neg_subj_np,
            neg_obj_np = self.neg_obj_np,
            neg_subj_neigh_np = self.neg_subj_neigh_np,
            neg_obj_neigh_np = self.neg_obj_neigh_np,
        )
        
        print(f'Dumped to {fpath} successfully!')
        return None
    
    def load(self, fpaths:list):
        temp_np_objs = []
        
        print_names = '\n'.join(['\t' + fp.name for fp in fpaths])
        
        print(f'Attempting to load:\n{print_names} \n')
        
        pbar = tqdm(fpaths, desc='Loading np tables')
        for fpath in pbar:
          
            fpath = Path(fpath)
            #fpath.mkdir(parents=True, exist_ok=True)
            
            pbar.set_description(f'Current ckpt: {fpath.name}')
            
            assert fpath.is_dir()
            assert fpath.joinpath('lists.json').is_file()
            assert fpath.joinpath('arrays.npz').is_file()

            with open(fpath.joinpath('lists.json'), 'r') as f:
                temp_json_file = json.load(f)

                self.positive_predicate_list += temp_json_file['positive_predicate_list']
                self.negative_predicates_list += temp_json_file['negative_predicates_list']

            temp_np_obj = np.load(fpath.joinpath('arrays.npz'))
            temp_np_objs.append(temp_np_obj)
            
        #if self.pos_subj_np is None:
        #    self.pos_subj_np = temp_np_obj['pos_subj_np']
        #    self.pos_obj_np = temp_np_obj['pos_obj_np']
        #    self.pos_subj_neigh_np = temp_np_obj['pos_subj_neigh_np']
        #    self.pos_obj_neigh_np = temp_np_obj['pos_obj_neigh_np']
        #    self.neg_subj_np = temp_np_obj['neg_subj_np']
        #    self.neg_obj_np = temp_np_obj['neg_obj_np']
        #    self.neg_subj_neigh_np = temp_np_obj['neg_subj_neigh_np']
        #    self.neg_obj_neigh_np = temp_np_obj['neg_obj_neigh_np']
        
        print()
        print('Concatenating...')
        
        self.pos_subj_np = np.concatenate(
          ([k['pos_subj_np'] for k in temp_np_objs])
        )
        self.pos_obj_np = np.concatenate(
          ([k['pos_obj_np'] for k in temp_np_objs])
        )
        self.pos_subj_neigh_np = np.concatenate(
          ([k['pos_subj_neigh_np'] for k in temp_np_objs])
        )
        self.pos_obj_neigh_np = np.concatenate(
          ([k['pos_obj_neigh_np'] for k in temp_np_objs])
        )
        self.neg_subj_np = np.concatenate(
          ([k['neg_subj_np'] for k in temp_np_objs])
        )
        self.neg_obj_np = np.concatenate(
          ([k['neg_obj_np'] for k in temp_np_objs])
        )
        self.neg_subj_neigh_np = np.concatenate(
          ([k['neg_subj_neigh_np'] for k in temp_np_objs])
        )
        self.neg_obj_neigh_np = np.concatenate(
          ([k['neg_obj_neigh_np'] for k in temp_np_objs])
        )
        
        self.get_size_params()
        self.Construct_dataloader_np_dict()
        
        return None
    
    def __len__(self):
        return len(self.positive_predicate_list)
    
    def __getitem__(self, idx:int) -> dict:
        
        item = self.emb_one_sample(
          self.get_PredEmb_item(
            self.np_to_sample(idx)
          )
        )
        
        return item