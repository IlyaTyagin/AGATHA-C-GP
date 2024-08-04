from collections import defaultdict
import torch
from agatha.ml.module import AgathaModule
from argparse import Namespace, ArgumentParser
from agatha.ml.util.lamb_optimizer import Lamb
from typing import List, Tuple, Dict, Any, Optional
from agatha.util.entity_types import is_umls_term_type, is_predicate_type
from agatha.util.sqlite3_lookup import Sqlite3Graph
from agatha.ml.util.embedding_lookup import EmbeddingLookupTable
from agatha.util.misc_util import iter_to_batches
from agatha.ml.hypothesis_predictor import predicate_util
from pathlib import Path
import os
from agatha.ml.util import hparam_util
import random

from agatha.util.matrix_lookup import np_emb_lookup_table, np_graph, np_emb_lookup_chunked_table
import json
import scipy
from tqdm import tqdm


class HypothesisPredictor(AgathaModule):
  def __init__(self, hparams:Namespace):
    super(HypothesisPredictor, self).__init__(hparams)

    # If the hparams have been setup with paths, typical for training
    
    #print(hparams)
    #print(100*'-')
    
    try:
        self.default_root_dir = hparams.default_root_dir
    except:
        pass
    
    if (hasattr(hparams, "load_emb_to_ram")):
        if hparams.load_emb_to_ram:
            self.load_emb_to_ram = True
        else:
            self.load_emb_to_ram = False
    else:
        self.load_emb_to_ram = False
    
    if (hasattr(hparams, "virt_adj_list")):
        if hparams.virt_adj_list:
            self.virt_adj_list = hparams.virt_adj_list
        else:
            self.virt_adj_list = None
    else:
        self.virt_adj_list = None
    
    if (hasattr(hparams, "neg_subsample_list")):
        if hparams.neg_subsample_list:
            self.neg_subsample_list = hparams.neg_subsample_list
        else:
            self.neg_subsample_list = None
    else:
        self.neg_subsample_list = None
    
    if (hasattr(hparams, "top_k_sim_dict_fpath")): 
        if hparams.top_k_sim_dict_fpath:
            self.top_k_sim_dict_fpath = hparams.top_k_sim_dict_fpath
        else:
            self.top_k_sim_dict_fpath = None
    else:
        self.top_k_sim_dict_fpath = None
        
    if not (hasattr(hparams, "subj_neighbor_sample_rate")):
        hparams.subj_neighbor_sample_rate = hparams.neighbor_sample_rate
    
    if (hasattr(hparams, "umls_to_st_dict_path")):
        #print('YES ST DICT')
        self.umls_to_st_dict_path_int = hparams.umls_to_st_dict_path
    else:
        #print('CANT FIND ST DICT')
        self.umls_to_st_dict_path_int = None
    
    
    self.graph = None
    self.embeddings = None
    if (
        hasattr(hparams, "graph_db")
        and hasattr(hparams, "entity_db")
        and hasattr(hparams, "embedding_dir")
    ):
      self.configure_paths(
          graph_db=hparams.graph_db,
          entity_db=hparams.entity_db,
          embedding_dir=hparams.embedding_dir,
          disable_cache=hparams.disable_cache,
      )

    # Backwards compatability for `simple` kwarg
    if not hasattr(self.hparams, "simple"):
      self.hparams.simple = False
    elif self.hparams.simple:
      # If we're using the simple model, we're going to overwrite the
      # neighbor_sample_rate
      self.hparams.neighbor_sample_rate = 0

    # Layers
    ## Graph Emb Input
    self.embedding_transformation = torch.nn.Linear(
        self.hparams.dim, self.hparams.dim
    )
    if self.hparams.simple:
      # Take two embeddings, produce one value
      self.simple_linear = torch.nn.Linear(2*self.hparams.dim, 1)
    else:
      ## Encoder Stack
      self.encode_predicate_data = torch.nn.TransformerEncoder(
          encoder_layer=torch.nn.TransformerEncoderLayer(
            self.hparams.dim,
            self.hparams.transformer_heads,
            self.hparams.transformer_ff_dim,
            self.hparams.transformer_dropout,
            "relu"
          ),
          num_layers=self.hparams.transformer_layers,
          norm=torch.nn.LayerNorm(self.hparams.dim),
      )
    ## Avg emb to logit
    self.encoding_to_logit = torch.nn.Linear(
      self.hparams.dim, 1
    )
    # Loss Fn
    self.loss_fn = torch.nn.MarginRankingLoss(margin=self.hparams.margin)

    # Helper data, set by prepare_for_training
    self.training_examples = None
    self.training_predicates = None
    self.validation_examples = None
    self.validation_predicates = None
    
    self.predicates = None
    self.coded_terms = None
    self.pairs_set = set()
    
    if (hasattr(hparams, "pos_samples_list")):
      if hparams.pos_samples_list:
        print(f'\t\tUsing training predicates from: {hparams.pos_samples_list}')
        with open(hparams.pos_samples_list, 'r') as f:
          self.predicates = json.load(f)
        
          self.pairs_set = {
              predicate_util.parse_predicate_name(predicate) for predicate in tqdm(
                  self.predicates, desc='Parsing positive pairs...'
              )
          }
        
    
    ## numpy cache directory, add argparse interface later
    
    self.np_cache_dir = Path(
      '/work/acslab/users/tyagin/Agatha_data/moe_training/prefetch/aapp_aapp/cache_np'
    )
    self.np_cache_dir = None
    
    if self.np_cache_dir:
      self.use_np_cache = True
      self.train_cache_sample_size = 40
      self.val_cache_sample_size = 1
    else:
      self.use_np_cache = False

  def configure_paths(
      self,
      graph_db:Path,
      entity_db:Path,
      embedding_dir:Path,
      disable_cache:bool=False
  ):
    
    path_separator = '|'
    entity_db_sqlite = None
    entity_db_split = str(entity_db).split(path_separator)
    entity_db_json = Path(entity_db_split[0])
    if len(entity_db_split) > 1:
        entity_db_sqlite = Path(entity_db_split[1])
    
    graph_db = Path(graph_db)
    #entity_db = Path(entity_db)
    embedding_dir = Path(embedding_dir)
    
    
    with open(entity_db_json, 'r') as f:
        nodelbl_to_int_id_dict = json.load(f)
        
    if self.virt_adj_list:
        print(f'Using virtual adjacency list from: {self.virt_adj_list}')
        virt_nodes_adj_list_dict = json.load(
            open(self.virt_adj_list)
        )
    else:
        virt_nodes_adj_list_dict = None
    
    if entity_db_sqlite:
        print(f'\t\tUsing PTBG embeddings from: {entity_db_sqlite}')
        self.embeddings = EmbeddingLookupTable(
            embedding_dir=embedding_dir,
            entity_db=entity_db_sqlite,
            disable_cache=True,
        )
    else:
        use_mmap_flag = not self.load_emb_to_ram
        print(f'\t\tUsing numpy stuff from: {entity_db_json.name}, use memmap: {use_mmap_flag}')
        self.embeddings = np_emb_lookup_table(
            nodelbl_to_int_id_dict=nodelbl_to_int_id_dict,
            emb_path=embedding_dir,
            memmap=use_mmap_flag,
            virt_nodes_adj_list=virt_nodes_adj_list_dict
        )
        
        # self.embeddings = np_emb_lookup_chunked_table(
        #     #nodelbl_to_int_id_dict=nodelbl_to_int_id_dict,
        #     #emb_path=embedding_dir,
        #     emb_w_ids_fpath='/lustre/scratch/acslab/agatha_2021_11_22_data/2021_11_22_w_semnet/2021_11_22_w_semnet_pred_emb',
        #     memmap=use_mmap_flag,
        #     #virt_nodes_adj_list=virt_nodes_adj_list_dict
        # )

    
    #self.graph=Sqlite3Graph(
    #    graph_db,
    #    disable_cache=disable_cache,
    #)
    
    #scp_adj_path = Path(
    #    '/lustre/acslab/shared/Agatha_shared/030122/ensemble/'
    #    'adj_matrices/scipy_csr/aapp_gngm_csr.npz'
    #)
    scp_adj_path = graph_db
    scp_adj_matr = scipy.sparse.load_npz(scp_adj_path)
    
    self.graph=np_graph(
        nodelbl_to_int_id_dict=nodelbl_to_int_id_dict,
        csr_adj_matr=scp_adj_matr,
        #filter_node_types='mp',
        virt_nodes_adj_list=virt_nodes_adj_list_dict
    )

  def paths_set(self)->bool:
    return self.embeddings is not None and self.graph is not None

  def _assert_configured(self)->None:
    assert self.paths_set(), \
      "You must call `model.configure_paths(...)` before running the model."

  def predict_from_terms(
      self,
      terms:List[Tuple[str, str]],
      batch_size:int=1,
      show_progress=False,
      subj_neighbor_sample_rate=None,
  )->List[float]:
    """Evaluates the Agatha model for the given set of predicates.

    For each pair of coded terms in `terms`, we produce a prediction in the
    range 0-1. Behind the scenes this means that we will lookup embeddings for
    the terms themselves as well as samples neighbors of each term. Then, these
    samples will be put through the Agatha transformer model to output a
    ranking criteria in 0-1. If this model has been put on gpu with a command
    like `model.cuda()`, then these predictions will happen on GPU. We will
    batch the predictions according to `batch_size`. This can greatly increase
    performance for large prediction sets.

    Note, behind the scenes there a lot of database accesses and caching. This
    means that your first calls to predict_from_terms will be slow. If you want
    to make many predictions quickly, call `model.preload()` before this
    function.

    Example Usage:

    ```python3
    model = torch.load(...)
    model.configure_paths(...)
    model.predict_from_terms([("C0006826", "C0040329")])
    > [0.9951196908950806]
    ```

    Args:
      terms: A list of coded-term name pairs. Coded terms are any elements that
        agatha names with the `m:` prefix. The prefix is optional when specifying
        terms for this function, meaning "C0040329" and "m:c0040329" will both
        correspond to the same embedding.
      batch_size: The number of predicates to predict at once. This is
        especially important when using the GPU.

    Returns:
      A list of prediction values in the `[0,1]` interval. Higher values
      indicate more plausible results. Output `i` corresponds to `terms[i]`.

    """
    self._assert_configured()
    
    if not subj_neighbor_sample_rate:
        subj_neighbor_sample_rate = self.hparams.neighbor_sample_rate
    # This will formulate our input as PredicateEmbeddings examples.
    observation_generator = predicate_util.PredicateObservationGenerator(
        graph=self.graph,
        embeddings=self.embeddings,
        neighbor_sample_rate=self.hparams.neighbor_sample_rate,
        subj_neighbor_sample_rate=subj_neighbor_sample_rate,
    )
    #print(terms)
    # Clean all of the input terms
    predicates = [
        predicate_util.to_predicate_name(
          predicate_util.clean_coded_term(s),
          predicate_util.clean_coded_term(o),
        )
        for s, o in terms
    ]

    result = []
    for predicate_batch in tqdm(
      iter_to_batches(predicates, batch_size),
      disable=not show_progress,
      desc='Batches progress',
      total=int(len(predicates)/batch_size) + 1,
    ):
      # Get a tensor representing each stacked sample
      batch = predicate_util.collate_predicate_embeddings(
          [observation_generator[p] for p in predicate_batch]
      )
      # Move batch to device
      batch = batch.to(self.get_device())
      result += self.forward(batch).detach().cpu().numpy().tolist()
    return result

  def predict_from_terms_latest_layer(
      self,
      terms:List[Tuple[str, str]],
      batch_size:int=1,
      show_progress=False,
  )->List[float]:

    self._assert_configured()
    # This will formulate our input as PredicateEmbeddings examples.
    observation_generator = predicate_util.PredicateObservationGenerator(
        graph=self.graph,
        embeddings=self.embeddings,
        neighbor_sample_rate=self.hparams.neighbor_sample_rate,
    )
    # Clean all of the input terms
    predicates = [
        predicate_util.to_predicate_name(
          predicate_util.clean_coded_term(s),
          predicate_util.clean_coded_term(o),
        )
        for s, o in terms
    ]

    result = []
    result_last_layer = []
    for predicate_batch in tqdm(
      iter_to_batches(predicates, batch_size),
      disable=not show_progress,
      desc='Batches progress',
      total=int(len(predicates)/batch_size) + 1,
    ):
      # Get a tensor representing each stacked sample
      batch = predicate_util.collate_predicate_embeddings(
          [observation_generator[p] for p in predicate_batch]
      )
      # Move batch to device
      batch = batch.to(self.get_device())
      #result += self.forward(batch).detach().cpu().numpy().tolist()
      
      predicate_embeddings = batch
      # Size <seq_len> X <batch_size> X <dim>
      local_stacked_emb = self.embedding_transformation(predicate_embeddings)
      local_stacked_emb = torch.relu(local_stacked_emb)
      encoded_predicate = self.encode_predicate_data(local_stacked_emb)
      encoded_predicate = encoded_predicate.mean(dim=0)
      
      # saving predicate 
      result_last_layer += encoded_predicate.detach().cpu().numpy().tolist()
      
      # continuing to get the score 
      logit = self.encoding_to_logit(encoded_predicate)
      logit = torch.sigmoid(logit)
      result += logit.detach().cpu().numpy().tolist()
      
    return list(zip(result_last_layer, result))

  def predict_from_terms_ext(
      self,
      terms:List[Tuple[str, str]],
      batch_size:int=1,
  )->List[float]:

    self._assert_configured()
    # This will formulate our input as PredicateEmbeddings examples.
    observation_generator = predicate_util.PredicateObservationGenerator(
        graph=self.graph,
        embeddings=self.embeddings,
        neighbor_sample_rate=self.hparams.neighbor_sample_rate,
    )
    # Clean all of the input terms
    predicates = [
        predicate_util.to_predicate_name(
          predicate_util.clean_coded_term(s),
          predicate_util.clean_coded_term(o),
        )
        for s, o in terms
    ]

    result = []
    batch_list = []
    for predicate_batch in iter_to_batches(predicates, batch_size):
      # Get a tensor representing each stacked sample
      batch = predicate_util.collate_predicate_embeddings(
          [observation_generator[p] for p in predicate_batch]
      )
      # Move batch to device
      batch_list.append(batch.to('cpu'))
      batch = batch.to(self.get_device())
      result += self.forward(batch).detach().cpu().numpy().tolist()
    
    return torch.stack(batch_list), result

  def preload(self, include_embeddings:bool=False)->None:
    """Loads all supplemental information into memory.

    The graph and entity databases as well as the set of embedding file are all
    read from storage in the typical case. If `model.preload()` is called, then
    the databases are loaded to memory, which can improve overall training
    performance. We do not preload the embedding by default because the
    EmbeddingLookupTable will automatically cache needed embedding files in a
    lazy manner. If we want to load these embeddings up front, we can set
    `include_embeddings`.

    Args:
      include_embeddings: If set, load all embedding files up front.

    """
    self._assert_configured()
    if not self.is_preloaded():
      self.graph.preload()
      if include_embeddings:
        self.embeddings.preload()
      else:
        self.embeddings.entities.preload()

  def is_preloaded(self)->None:
    return (
        self.paths_set()
        and self.graph.is_preloaded()
        and self.embeddings.entities.is_preloaded()
    )

  def prepare_for_training(self)->None:
    self._assert_configured()
    
    if not self.use_np_cache:
    
      entities = self.embeddings.keys()
      assert len(entities) > 0, "Failed to find embedding entities."
      self.coded_terms = list(filter(is_umls_term_type, entities))
      
      if not self.predicates:
        self.predicates = list(filter(
          predicate_util.is_valid_predicate_name,
          entities
        ))
      if self.neg_subsample_list:
          print(f'Using specifically curated set of negatives from: {self.neg_subsample_list}')
          self.neg_subsample_list = json.load(open(self.neg_subsample_list))
      else:
          self.neg_subsample_list = None
    
      if self.top_k_sim_dict_fpath:
          print(f'Using similarity-based negatives from: {self.top_k_sim_dict_fpath}')
          self.top_k_sim_dict = json.load(open(self.top_k_sim_dict_fpath))
      else:
          self.top_k_sim_dict = None
        
      self.training_predicates, self.validation_predicates = \
          self.training_validation_split(self.predicates)
      self.training_examples = predicate_util.PredicateExampleDataset(
          predicate_ds=self.training_predicates,
          all_predicates=self.predicates,
          embeddings=self.embeddings,
          graph=self.graph,
          coded_terms=self.coded_terms,
          neg_subsamples=self.neg_subsample_list,
          pos_pairs=self.pairs_set,
          top_k_sim_dict=self.top_k_sim_dict,
          neighbor_sample_rate=self.hparams.neighbor_sample_rate,
          subj_neighbor_sample_rate=self.hparams.subj_neighbor_sample_rate,
          negative_swap_rate=self.hparams.negative_swap_rate,
          negative_scramble_rate=self.hparams.negative_scramble_rate,
          negative_sts_rate=self.hparams.negative_sts_rate,
          negative_subset_rate=self.hparams.negative_subset_rate,
          negative_topk_rate=self.hparams.negative_topk_sim_rate,
          preload_on_first_call=not self.hparams.disable_cache,
          umls_to_st_dict_path=self.umls_to_st_dict_path_int,
      )

      self.validation_examples = predicate_util.PredicateExampleDataset(
          predicate_ds=self.validation_predicates,
          all_predicates=self.predicates,
          embeddings=self.embeddings,
          graph=self.graph,
          coded_terms=self.coded_terms,
          neg_subsamples=self.neg_subsample_list,
          pos_pairs=self.pairs_set,
          top_k_sim_dict=self.top_k_sim_dict,
          neighbor_sample_rate=self.hparams.neighbor_sample_rate,
          subj_neighbor_sample_rate=self.hparams.subj_neighbor_sample_rate,
          negative_swap_rate=self.hparams.negative_swap_rate,
          negative_scramble_rate=self.hparams.negative_scramble_rate,
          negative_sts_rate=self.hparams.negative_sts_rate,
          negative_subset_rate=self.hparams.negative_subset_rate,
          negative_topk_rate=self.hparams.negative_topk_sim_rate,
          preload_on_first_call=not self.hparams.disable_cache,
          umls_to_st_dict_path=self.umls_to_st_dict_path_int,
      )
      self._vprint("Ready for training! (No cached links are used)")

  def train_dataloader(self)->torch.utils.data.DataLoader:
    self._vprint("Getting Training Dataloader")
    
    if self.use_np_cache:
      tdl_obj = predicate_util.Numpy_cache_links_obj_ens(embeddings=self.embeddings.emb_matrix)
      fnames_flist = list(self.np_cache_dir.joinpath('train_v2').glob('*/'))
      #print(fnames_flist)
      fnames_sample = random.sample(
        fnames_flist, 
        self.train_cache_sample_size
      )
      tdl_obj.load(fnames_sample)
      self.training_examples = tdl_obj
    
    return self._configure_dataloader(
        self.training_examples,
        shuffle=True,
        batch_size=self.hparams.positives_per_batch,
        collate_fn=predicate_util.collate_predicate_training_examples,
    )

  def val_dataloader(self)->torch.utils.data.DataLoader:
    self._vprint("Getting Validation Dataloader")
    
    if self.use_np_cache:
      tdl_obj = predicate_util.Numpy_cache_links_obj_ens(embeddings=self.embeddings.emb_matrix)
      fnames_flist = list(self.np_cache_dir.joinpath('train_v2').glob('*/'))
      fnames_sample = random.sample(
        fnames_flist, 
        self.val_cache_sample_size
      )
      tdl_obj.load(fnames_sample)
      self.validation_examples = tdl_obj
    
    return self._configure_dataloader(
        self.validation_examples,
        shuffle=False,
        batch_size=self.hparams.positives_per_batch,
        collate_fn=predicate_util.collate_predicate_training_examples,
    )

  def forward(self, predicate_embeddings:torch.FloatTensor)->torch.FloatTensor:
    #print('Calling model.forward()')
    # Size <seq_len> X <batch_size> X <dim>
    local_stacked_emb = self.embedding_transformation(predicate_embeddings)
    local_stacked_emb = torch.relu(local_stacked_emb)
    if hasattr(self.hparams, "simple") and self.hparams.simple:
      # Reorder to <batch_size> X <seq_len> X <dim>
      # Flatten to <batch_size> X <dim*seq_len> (in this case seq_len=2)
      logit = self.simple_linear(local_stacked_emb.permute(1, 0, 2).flatten(1))
    else:
      encoded_predicate = self.encode_predicate_data(local_stacked_emb)
      encoded_predicate = encoded_predicate.mean(dim=0)
      logit = self.encoding_to_logit(encoded_predicate)
    logit = torch.sigmoid(logit)
    return logit.reshape(-1)

  def _step(
      self,
      positive_predicates:List[str],
      positive_observations:torch.FloatTensor,
      negative_predicates_list:List[List[str]],
      negative_observations_list:List[torch.FloatTensor],
  )->Tuple[torch.Tensor, Dict[str, Any]]:
    """ Performs a forward pass of the model during training.

    Used in both training_step and validation_step, this function accepts a set
    of predicate names and performs a forward pass of the hypothesis generation
    training routine. This involves generating negative samples for each
    positive example and evaluating metrics that quantify the difference
    between the two.

    Args:
      positive_predicates: List of string names associated with positive batch
      positive_observations: Packed tensor containing info corresponding to
        positive_predicates. Shape: <seq_len> X <batch_size> X <dim>
      negative_predicates_list: List of predicate lists.
        negative_predicates_list[i] corresponds to the i'th negative sample
        batch. Each batch should be the same size as the positive batch.
      negative_observations_list: List of packed tensors.
        negative_observations_list[i] corresponds to the i'th negative sample.

    Returns:
      The first element is the loss tensor, used for back propagation. The
      second element is a dict containing all extra metrics.

    """
    # Do positive checks
    assert isinstance(positive_observations, torch.Tensor), \
      f"Err: positive_observations is {type(positive_observations)}"
    assert len(positive_observations.shape) == 3
    _, actual_batch_size, actual_dim = positive_observations.shape
    assert len(positive_predicates) == actual_batch_size
    assert self.hparams.positives_per_batch == actual_batch_size
    assert self.hparams.dim == actual_dim

    # no negative checks
    assert len(negative_predicates_list) == len(negative_observations_list)
    for n_preds, n_obs in zip(negative_predicates_list, negative_observations_list):
      assert isinstance(n_obs, torch.Tensor)
      assert len(n_obs.shape) == 3
      _, actual_batch_size, actual_dim = n_obs.shape
      assert len(n_preds) == actual_batch_size
      assert self.hparams.positives_per_batch == actual_batch_size
      assert self.hparams.dim == actual_dim

    positive_predictions = self.forward(positive_observations)
    # We cannot tolerate an error on a positive sample
    # An error occurs if any positive prediction is _not_ finite
    # Note that `~` is bitwise "not" for our boolean matrix
    if torch.any(~torch.isfinite(positive_predictions.detach().cpu())):
      print(positive_predicates)
      raise ValueError("Invalid positive sample")

    partial_losses = []
    for negative_predicates, negative_observations in zip(
        negative_predicates_list, negative_observations_list
    ):
      negative_predictions = self.forward(negative_observations)
      # We CAN tolerate an error on a negative sample
      if torch.any(~torch.isfinite(negative_predictions.detach().cpu())):
        # print debug info
        print("ERROR: Encountered an issue with a negative predicate:")
        print("Negative Predicate Scores:")
        print(negative_predictions)
        print("Negative Predicates")
        print(negative_predicates)
      else:
        partial_losses.append(
            self.loss_fn(
              positive_predictions,
              negative_predictions,
              positive_predictions.new_ones(len(positive_predictions))
            )
        )
    assert len(partial_losses) > 0, "Failure occurred on all negative batches."
    # End of batch
    loss=torch.mean(torch.stack(partial_losses))
    return (
        loss,
        dict( # pbar metrics
      )
    )

  def training_step(
      self,
      inputs:Dict[str, Any],
      batch_idx:int
  )->Dict[str, Any]:
    """
    The input to this function is the output of
    collate_predicate_training_examples
    """
    loss, metrics = self._step(**inputs)
    
    pb_metrics = {'train_loss': "{:.9f}".format(loss)}
    
    return dict(
        loss=loss,
        progress_bar=pb_metrics,
        log=dict(train_loss=loss)
    )

  def validation_step(
      self,
      inputs:Dict[str, Any],
      batch_idx:int
  )->Dict[str, Any]:
    """
    The input to this function is the output of
    collate_predicate_training_examples
    """
    loss, metrics = self._step(**inputs)
    val_metrics = {f"val_{k}": v for k, v in metrics.items()}
    val_metrics["val_loss"] = loss
    return val_metrics

  def configure_optimizers(self):
    self._vprint("Configuring optimizers")
    return Lamb(
        self.parameters(),
        lr=self.hparams.lr,
        weight_decay=self.hparams.weight_decay,
    )

  def optimizer_step(
      self,
      epoch_idx,
      batch_idx,
      optimizer,
      optimizer_idx,
      second_order_closure=None
  ):
    # warm up lr
    if  self.trainer.global_step < self.hparams.warmup_steps:
      lr_scale = min(
          1.,
          float(self.trainer.global_step + 1)/float(self.hparams.warmup_steps)
      )
      for pg in optimizer.param_groups:
        pg['lr'] = lr_scale * self.hparams.lr
    optimizer.step()
    optimizer.zero_grad()
    
    return None
    
#   def training_epoch_end(self, outputs) -> None:
#     print('Saving model at the epoch end')
#     # model_save_dir = Path(Path(self.default_root_dir)) / 'saved_model_ckpts'
#     # model_save_dir.mkdir(exist_ok=True, parents=True)
#     # model_save_fname = model_save_dir / f'epoch_{self.current_epoch}.cpkt'
#     # #print(dir(self))
#     # torch.save(
#     #     self.state_dict(),
#     #     model_save_fname
#     # )
    
#     #self.validation_epoch_end(outputs)
#     self._on_epoch_end(outputs)
    
#     return dict()


  @staticmethod
  def add_argparse_args(parser:ArgumentParser)->ArgumentParser:
    """Used to add all model parameters to argparse

    This static function allows for the easy configuration of argparse for the
    construction and training of the Agatha deep learning model. Example usage:

    ```python3
    parser = HypothesisPredictor.add_argparse_args(ArgumentParser())
    args = parser.parse_args()
    trainer = Trainer.from_argparse_args(args)
    model = HypothesisPredictor(args)
    ```

    Note, many of the arguments, such as the location of training databases or
    the paths used to save the model during training, will _NOT_ be serialized
    with the model. These can be configured either from `args` directly after
    parsing, or through `configure_paths` after training.

    Args:
      parser: An argparse parser to be configured. Will receive all necessary
        training and model parameter flags.

    Returns:
      A reference to the input argument parser.

    """
    parser = AgathaModule.add_argparse_args(parser)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--entity-db", type=Path)
    parser.add_argument("--graph-db", type=Path)
    parser.add_argument("--nodeid-mappings", type=Path)
    parser.add_argument("--pos-samples-list", type=Path)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--negative-scramble-rate", type=int)
    parser.add_argument("--negative-swap-rate", type=int)
    parser.add_argument("--negative-sts-rate", type=int)
    parser.add_argument("--negative-subset-rate", type=int, default=0)
    parser.add_argument("--negative-topk-sim-rate", type=int, default=0)
    parser.add_argument("--neighbor-sample-rate", type=int)
    parser.add_argument("--positives-per-batch", type=int)
    parser.add_argument("--transformer-dropout", type=float)
    parser.add_argument("--transformer-ff-dim", type=int)
    parser.add_argument("--transformer-heads", type=int)
    parser.add_argument("--transformer-layers", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--umls-to-st-dict-path", type=Path)
    parser.add_argument("--load-emb-to-ram", action="store_true")
    parser.add_argument("--subj-neighbor-sample-rate", type=int, default=15)
    parser.add_argument("--virt-adj-list", type=Path, default=None)
    parser.add_argument("--neg-subsample-list", type=Path, default=None)
    parser.add_argument("--top-k-sim-dict-fpath", type=Path, default=None)
    parser.add_argument(
        "--simple",
        help="If set, ignore graph and use a simpler model architecture.",
        action="store_true"
    )
    return parser
