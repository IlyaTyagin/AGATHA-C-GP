from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from joblib import Parallel, delayed

class np_emb_lookup_table():
    
    def __init__(
        self,
        nodelbl_to_int_id_dict:dict,
        emb_path,
        memmap=False,
        virt_nodes_adj_list=None,
    ):
        self.memmap = None
        if memmap:
            self.memmap = 'r'
            print('Memmap is used, embeddings are NOT loaded to RAM.')
        else:
            print('Memmap is not used, embeddings are loaded to RAM.')
        
        self.emb_path = Path(emb_path)
        
        self.nodelbl_to_int_id_dict = None
        self.int_id_to_node_lbl_arr = None
        
        self.virt_nodes_adj_list = virt_nodes_adj_list
        
        self.nodelbl_to_int_id_dict = nodelbl_to_int_id_dict
        self.int_id_to_node_lbl_arr = list(self.nodelbl_to_int_id_dict)
        
        self.emb_matrix = np.load(self.emb_path, mmap_mode=self.memmap)
        
        # assert self.emb_matrix.shape[0] == len(self.int_id_to_node_lbl_arr)
        
        if self.virt_nodes_adj_list:
            self._getitem_impl = self.getitem_w_virt_agg
        else:
            self._getitem_impl = self.getitem_reg
        
        return None
    
    def __getitem__(self, nodeid):
        # Delegate to the chosen method
        return self._getitem_impl(nodeid) 
    
    def getitem_reg(self, nodeid):
        #print(nodeid)
        idx = self.nodelbl_to_int_id_dict[nodeid]
        
        return np.array(self.emb_matrix[idx])

    def getitem_w_virt_agg(self, nodeid):
        #print(nodeid)
        idx = self.nodelbl_to_int_id_dict.get(nodeid)
        
        if idx is not None:
            emb = np.array(self.emb_matrix[idx])
        else:
            node_neig_list = self.virt_nodes_adj_list[nodeid]
            node_neig_emb_list = []
            
            for n in node_neig_list:
                node_neig_emb_list.append(
                    self.getitem_reg(n)
                )
            
            emb = np.mean(node_neig_emb_list, axis=0)
        
        return emb
    
    def __len__(self):
        return self.int_id_to_node_lbl_arr.shape[0]
    
    def keys(self):
        return self.nodelbl_to_int_id_dict.keys()
    
    def preload(self):
        pass

class np_graph():
    
    def __init__(
        self,
        nodelbl_to_int_id_dict:dict,
        csr_adj_matr,
        memmap=True,
        filter_node_types:str='',
        virt_nodes_adj_list=None,
    ):
        self.memmap = None
        if memmap:
            self.memmap = 'r'
            
        self.csr_adj_matr = csr_adj_matr
        self.nodelbl_to_int_id_dict = nodelbl_to_int_id_dict
        self.int_id_to_node_lbl_arr = list(self.nodelbl_to_int_id_dict)
        self.node_type_filter = filter_node_types
        
        self.virt_nodes_adj_list = virt_nodes_adj_list
        
        if self.node_type_filter:
            self.filter_nodeids()
            
        if self.virt_nodes_adj_list:
            self._getitem_impl = self.getitem_w_virt_agg
        else:
            self._getitem_impl = self.getitem_reg
        
        return None
    
    def __getitem__(self, nodeid):
        # Delegate to the chosen method
        return self._getitem_impl(nodeid)
    
    def getitem_reg(self, nodeid):
        idx = self.nodelbl_to_int_id_dict[nodeid]
        
        neigh_idxs = self.get_neigh_idsx(idx)
        
        return self.neigh_idsx_to_nodeids(neigh_idxs)
    
    def getitem_w_virt_agg(self, nodeid):
        if nodeid in self.nodelbl_to_int_id_dict:
            return self.getitem_reg(nodeid)
        else:
            return self.virt_nodes_adj_list[nodeid]
            
    def get_batch(self, nodeids):
        idxs = list(
            map(
                self.nodelbl_to_int_id_dict.__getitem__,
                nodeids
            )
        )
        
        neigh_idxs = self.get_neigh_idsx_batch(idxs)
        return self.neigh_idsx_to_nodeids(neigh_idxs)
    
    def filter_nodeids(self):
        self.node_type_filter = set(self.node_type_filter)
        print(f'Filtering nodes keeping only: {self.node_type_filter}')
        if 'p' in self.node_type_filter and 'm' in self.node_type_filter:
            pred_nodes = (
                {
                    k:v for k,v in tqdm(self.nodelbl_to_int_id_dict.items())
                    if k[0] == 'p'
                }
            )
            
            pred_nodes_mo_dict = dict()
            print('Keeping only CORRECT mesh terms')
            for pred in tqdm(pred_nodes):
                pred_split = pred.split(':')
                subj = pred[1] 
                obj = pred[-1]
                pred_nodes_mo_dict[pred] = self.nodelbl_to_int_id_dict[pred]
                pred_nodes_mo_dict[subj] = self.nodelbl_to_int_id_dict[subj]
                pred_nodes_mo_dict[obj] = self.nodelbl_to_int_id_dict[obj]
            
            self.nodelbl_to_int_id_dict = pred_nodes_mo_dict
            
            
        else:    
            self.nodelbl_to_int_id_dict = (
                {
                    k:v for k,v in tqdm(self.nodelbl_to_int_id_dict.items())
                    if k[0] in self.node_type_filter
                }
            )
    
    def get_neigh_idsx(self, idx):
        return self.csr_adj_matr[idx].nonzero()[1]
    
    def get_neigh_idsx_batch(self, idx_batch):
        return self.csr_adj_matr[idx_batch, :].sum(axis=0).nonzero()[1]
    
    def neigh_idsx_to_nodeids(self, idxs):
        
        nodeids_list = list(
            map(
                self.int_id_to_node_lbl_arr.__getitem__,
                idxs
            )
        )
        
        if self.node_type_filter:
            nodeids_list = (
                [
                    nodeid for nodeid in nodeids_list
                    if nodeid[0] in self.node_type_filter
                ]
            )
            
        return nodeids_list
    
    def __len__(self):
        return len(self.nodelbl_to_int_id_dict)
    
    def keys(self):
        return self.nodelbl_to_int_id_dict.keys()
    
    def __contains__(self, key):
        value_or_none = self.nodelbl_to_int_id_dict.get(key)
        
        in_virt_nodes_adj_list = False
        if self.virt_nodes_adj_list:
            in_virt_nodes_adj_list = key in self.virt_nodes_adj_list
        return (value_or_none is not None) or in_virt_nodes_adj_list
    
    def __iter__(self):
        return iter(self.nodelbl_to_int_id_dict)
    
    def preload(self):
        pass
    

class np_emb_lookup_chunked_table:
    def __init__(
        self,
        emb_w_ids_fpath,
        json_read_n_jobs=1,
        use_only_m_emb=False,
        memmap=True,
    ):
        
        self.memmap = None
        self.use_only_m_emb = use_only_m_emb
        if memmap:
            self.memmap = 'r'
            print('Memmap is used, CUI embeddings are not loaded to RAM.')
        else:
            print('Memmap is not used, CUI embeddings are loaded to RAM.')
        
        self.emb_w_ids_fpath = Path(emb_w_ids_fpath)
        
        self.pmids_chunk_list = None
        self.emb_chunks_dict = None
        self.pmids_to_loc_dict = None
        
        self.emb_chunk_np_flist = sorted(self.emb_w_ids_fpath.glob('*.npy'))
        self.pmids_chunk_json_flist = sorted(self.emb_w_ids_fpath.glob('chunk*.json'))
        
        # print(len(self.pmids_chunk_json_flist))
        # print()
        
        assert len(self.pmids_chunk_json_flist) == len(self.emb_chunk_np_flist)
        
        self.open_np_chunks()
        self.open_json_chunks(n_jobs=json_read_n_jobs)
        tqdm._instances.clear()
        self.construct_lookup_dict()
        
        return None
    
    def open_np_chunks(self):
        
        emb_chunks_dict = dict()

        for fname in tqdm(
            self.emb_chunk_np_flist,
            desc='Opening np chunks'
        ):
            #k = fname.stem.split('_')[-1]
            k = fname.stem.split('chunk_')[-1].split('__')[0]
            v = np.load(
                fname,
                mmap_mode=self.memmap,
            )
            emb_chunks_dict[k] = v
        
        self.emb_chunks_dict = emb_chunks_dict
        return None
    
    def open_json(self, fname):
        try:
            with open(fname, 'r') as f:
                js = json.load(f)
        except Exception as e:
            print(e)
            print(fname)
            return fname

        return js
    
    def open_json_chunks(self, n_jobs):
        self.pmids_chunk_list = (
            Parallel(n_jobs=n_jobs)(
                delayed(self.open_json)(fname) for fname in tqdm(
                    self.pmids_chunk_json_flist,
                    desc='Opening json index chunks'
                )
            )
        )
    
    def construct_lookup_dict(self):
        
        self.chunk_to_pmids_list_dict = dict(
            zip(
                [fname.stem.split('chunk_')[-1].split('__')[0] for fname in self.pmids_chunk_json_flist],
                self.pmids_chunk_list
            )
        )
        
        pmids_to_loc_dict = dict()
        
        if self.use_only_m_emb:
            print('Filtering out non-UMLS embeddings from index')
        
        for chunk_idx, pmids_list in tqdm(
            self.chunk_to_pmids_list_dict.items(),
            desc='Constructing CUI lookup index'
        ):
            for row_idx, pmid in enumerate(pmids_list):
                emb_loc = f'{chunk_idx}_{row_idx}'
                
                if self.use_only_m_emb and pmid[0] != 'm':
                    continue

                pmids_to_loc_dict[pmid] = emb_loc
        
        print(f'CUI index size: {len(pmids_to_loc_dict)}')
                
        self.pmids_to_loc_dict = pmids_to_loc_dict
        
        return None
    
    def __getitem__(self, pmid):
        
        emb_loc = self.pmids_to_loc_dict[pmid]
    
        chunk_idx, row_idx = emb_loc.split('_')

        emb_np = self.emb_chunks_dict[chunk_idx][int(row_idx)]

        return emb_np

    def __len__(self):
        return len(self.pmids_to_loc_dict)
    
    def __contains__(self, el):
        return el in self.pmids_to_loc_dict
    
    def keys(self):
        return self.pmids_to_loc_dict.keys()
    
    def preload(self):
        pass