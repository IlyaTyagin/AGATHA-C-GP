import re
import os
import subprocess
from pathlib import Path
import time
import numpy as np
import socket
import contextlib
from tqdm import tqdm
import joblib
from shutil import copyfile

import pickle
import json

from nltk.tokenize import sent_tokenize
from file_read_backwards import FileReadBackwards
from collections import defaultdict
from joblib import Parallel, delayed

import random

from typing import List, Any, Set, Optional, Tuple

import multiprocessing
#from agatha.construct.text_util import *

#from agatha.construct import construct_config_pb2 as cpb

class SemRepHandler():
    """Simple python interface for painless running SemRep tool.
    
    Features:
    - Uses temporary folder to handle queries:
        * from input strings
        * batch queries from .txt file
    - 
    
    """
    
    def _Stop_MM_services(self):
        "Stops MetaMap SKR and WSD services if they run."
        
        skr_proc = subprocess.run(f'{self.skr_file_path} stop', capture_output=True, shell=True)
        wsd_proc = subprocess.run(f'{self.wsd_file_path} stop', capture_output=True, shell=True)
        
        #print(output1, output2)
        
        if skr_proc.returncode == 0 and wsd_proc.returncode == 0:
            print('Existing services killed.')
        return
        
    
    def _Start_MM_services_(self) -> None:
        "starts MetaMap SKR and WSD services."
        
        output1 = subprocess.run(
            f'{self.skr_file_path} start', 
            capture_output=True, 
            shell=True
        )
        output2 = subprocess.Popen(
            f'{self.wsd_file_path} start', 
            shell=True,
        )
        time.sleep(30)
        
        
        # Check if services started
        server_names = ['MedPost-SKR', 'WSD_Server']
        current_java_procs = subprocess.run(
            'ps ax | grep java ', 
            shell=True, 
            capture_output=True
        ).stdout.decode('UTF-8')
        
        for server_name in server_names:
            if server_name not in current_java_procs:
                raise ProcessLookupError(
                    f'Process {server_name} did not start. '
                    'If you use a cluster, consider using <= 28 machines.'
                )
        
        print("\nSKR and WSD services started.")
        return
    
    def _handle_MM_services_(self) -> None:
        """Handles MM services. 
        Starts them if needed.
        """
        
        if self.restart_mm_services:
            self._Stop_MM_services()
            self._Start_MM_services_()
        else:
            server_names = ['MedPost-SKR', 'WSD_Server']
            current_java_procs = subprocess.run(
                'ps ax | grep java ', 
                shell=True, 
                capture_output=True
            ).stdout.decode('UTF-8')
            
            needs_start_MM = False
            for server_name in server_names:
                if server_name not in current_java_procs:
                    needs_start_MM = True
                    break
            
            if needs_start_MM:
                self._Stop_MM_services()
                self._Start_MM_services_()
        return
        
    def _create_folder_(self, path) -> None:
        if not path.exists():
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(e)
        return None 
    
    def _Create_temp_folders_(self) -> None:
        """"Handles temp folders behavior.
        
        We follow this strategy:
            Main folder: <temp_folder>
                Interactive file: <temp_folder>/temp_sr_input.txt
                Folder for each file: <temp_folder>/<filename>/
                SemRep log: <temp_folder>/sr_log.txt
                Files for temrorary splitting: <temp_folder>/splitTempFiles
                Partial SemRep results: <temp_folder>/semrepProcessed
        
        """
        self._create_folder_(self.temp_folder)
        
        return None
    
    
    def _perform_assertions_(self):
        """Checks if everything is OK with NLM soft file structure.
        
        """
        assert self.temp_folder.exists()
        
        assert self.nlm_soft_path.exists(), \
            "Please provide a valid path to NLM soft."
        
        assert self.skr_file_path.is_file()
        assert self.wsd_file_path.is_file()
        
        assert self.sr_binary_path.is_file(), \
            "SemRep binary is not found."
        return
    
    def __init__(
        self,
        nlm_soft_path:str,
        temp_folder:str,
        replace_utf8_path:str = '',
        debug:bool = True,
        restart_mm_services:bool = False,
        params:str = '',
    ):
        
            ### Path checking ###
        self.nlm_soft_path = Path(nlm_soft_path)
        self.temp_folder = Path(temp_folder)
        self.replace_utf8_path = Path(replace_utf8_path)
        
        self.sr_log_path = self.temp_folder.joinpath("sr_log.txt")
        self.skr_file_path = self.nlm_soft_path.joinpath("public_mm/bin/skrmedpostctl")
        self.wsd_file_path = self.nlm_soft_path.joinpath("public_mm/bin/wsdserverctl")
        self.sr_binary_path = self.nlm_soft_path.joinpath("public_semrep/bin/semrep.v1.9.BINARY.Linux")
        
            ### Additional parameters ###
        self.use_nltk_tokenizer = False
        self.restart_mm_services = restart_mm_services
        if not params:
            self.params = '--sldiID -DS -L 2021 -Z 2021AB -A -F -n --negex_st_add all -V NLM --relaxed_model'
        else:
            self.params = params
        
            ### Supplementary functions ###
        self._handle_MM_services_()
        self._Create_temp_folders_()
        self._perform_assertions_()
    
    def _write_input_file_(
        self,
        inputObj,
        file_path,
        replace_utf8:bool = False,
    ) -> None:
        """Creates input file for SemRep."""
        
        if type(inputObj) == dict:
            with open(file_path, 'w') as f:
                for k in inputObj:
                    if len(inputObj[k]) > 2:
                        f.write(f'{k}|{inputObj[k]}\n')
                    
        if type(inputObj) == list:
            with open(file_path, 'w') as f:
                for i, inputString in enumerate(inputObj):
                    f.write(f'user_input_{i}|{inputString}\n')
        
        if replace_utf8:
            assert self.replace_utf8_path.is_file(), \
                "Please provide a valid path to replace_utf8.jar file."
            print('Processing input with replace_utf8.jar utility...')
            orig_file_path = file_path.with_suffix('.orig')
            os.rename(file_path, orig_file_path)
            subprocess.run(
                f'''java -jar {self.replace_utf8_path} {orig_file_path} > {file_path}''',
                shell=True
            )
        
        return file_path
    
    def ProcessString(
        self, 
        inputList:list,
        raw:bool = False,
        replace_utf8:bool=True,
    ):
        
        "Handles interactive requests from Python."
        
        print('Starting SemRep in str mode...\n')
        
        temp_file_path = self.temp_folder.joinpath('temp_sr_input.txt')
        temp_out_file_path = self.temp_folder.joinpath('temp_sr_output.txt')
        
        self.Clean_temp_folder(verbose = False)
        
        #with open(temp_file_path, 'w') as f:
        #    for i, inputString in enumerate(inputList):
        #        f.write(f'user_input_{i}|{inputString}\n')
        
        self._write_input_file_(
            inputObj=inputList,
            file_path=temp_file_path,
            replace_utf8=replace_utf8,
        )
        
        #with open(temp_file_path) as f:
            #print(f.read())
        
        self._semrep_execute_(
            input_filename = temp_file_path, 
            output_filename = temp_out_file_path,
            parent_dir = self.temp_folder,
        )
        
        if raw:
            with open(temp_out_file_path) as f:
                parsed_sr_results = [
                    line for line in f.read().split('\n') if line
                ]
        else:
            parsed_sr_results = self.Parse_SemRep_results(temp_out_file_path)
        
        return parsed_sr_results
    
    def ProcessList_parallel(
        self,
        inputList:list=[],
        raw:bool = False,
        outputFilepath:str='',
        chunkSize:int = 1000,
        nthreads:int = -1,
        replace_utf8:bool = True,
        inputFilepath:str='',
        backend:str='loky',
        show_progress:bool=False,
        subfolder:str='abstracts',
        
    ) -> dict:
        """Handles parallel SemRep runs in interactive mode.
        
        Algorithm:
            1) Create a folder within <self.temp_folder> (e.g., with $HOSTNAME)
                * Example: temp_folder/node0092/
            2) Split list into <chunkSize> chunks
            3) Write temp files for each chunk
            4) For each chunk run SemRep using joblib with nthreads workers
        
        """

        
            ## Working with hostname ##
        try:
            worker_id = socket.gethostname().split('.')[0]
        except:
            worker_id = ''
        if not worker_id:
            random.seed(random.randint(10, 1000))
            worker_id = f'worker_{random.randint(1000, 9999)}'
        
        
        temp_parent_dir_path = self.temp_folder.joinpath(worker_id)
        temp_parent_dir_path_subfolder = temp_parent_dir_path.joinpath(subfolder)
        
        self.Clean_temp_folder(
            verbose = False,
            temp_path=temp_parent_dir_path_subfolder,
        )
        
        
        self._create_folder_(temp_parent_dir_path)
        self._create_folder_(temp_parent_dir_path_subfolder)
        
        temp_parent_dir_path = temp_parent_dir_path_subfolder
        
        temp_out_file_path = temp_parent_dir_path.joinpath('temp_sr_output.txt')
        
            ## Working with threads number ##
        
        if nthreads < 0:
            nthreads = int(multiprocessing.cpu_count()/2) - 1
            if nthreads < 1:
                nthreads = 1
        
            ## Splitting list into chunks ##
        
        temp_file_path = temp_parent_dir_path.joinpath('temp_sr_input.txt')
        
        #with open(temp_file_path, 'w') as f:
        #    for i, inputString in enumerate(inputList):
        #        f.write(f'user_input_{i}|{inputString}\n')
        
        if len(inputList) > 0 and len(str(inputFilepath)) == 0:
            print('Run SemRep in interactive mode...')
            self._write_input_file_(
                inputObj=inputList,
                file_path=temp_file_path,
                replace_utf8=replace_utf8,
            )
        elif len(inputList) == 0 and len(str(inputFilepath)) > 0:
            print('Run SemRep in file mode...')
            assert Path(inputFilepath).is_file(), \
                'Plese, provide a vaid input file path.'
            copyfile(inputFilepath, temp_file_path)
            print(f'Processing {inputFilepath}')
        elif len(inputList) == 0 and len(str(inputFilepath)) == 0:
            print(
                "No input provided. Nothing to compute. Returning empty dict..."
            )
            return dict()
        else:
            print(len(inputList), len(inputFilepath))
            raise ValueError(
                'Input list OR a filename containing input is expected.' 
                'Not both. Please, be specific about input.'
            ) 
        
        subprocess.run(
            f'''split {temp_file_path} -l {chunkSize} --numeric-suffixes {temp_file_path.with_suffix('')}_ --additional-suffix=.txt''', 
            shell=True)
        os.remove(temp_file_path)
        sr_input_filelist = [
            temp_parent_dir_path.joinpath(_) for _ in os.listdir(temp_parent_dir_path) if '.txt' in _]
        
        sr_output_filelist = [_.with_suffix('.out') for _ in sr_input_filelist]
        
        sr_parallel_args = list(zip(
            sr_input_filelist, 
            sr_output_filelist, 
            len(sr_input_filelist)*[temp_parent_dir_path]
        ))
        
            ## Joblib part ##
        
        #with self.tqdm_joblib(
        #    tqdm(
        #        desc=f'{worker_id} status', 
        #        total=len(sr_input_filelist),
        #    )
        #) as progress_bar:
        #    with Parallel(n_jobs = nthreads) as parallel:
        #        results_chunked = parallel(
        #            delayed(self._semrep_execute_)(*i) for i in sr_parallel_args)
        
        
        #with tqdm_joblib(
        #    tqdm(
        #        desc=f'{worker_id} status', 
        #        total=len(sr_input_filelist),
        #        position=0, 
        #        leave=True,
        #    )
        #) as progress_bar:
        #    results_chunked = Parallel(
        #        n_jobs = nthreads,
        #        backend=backend,
        #    )(
        #        delayed(self._semrep_execute_)(*i) for i in sr_parallel_args)
        
        if show_progress:
          sr_parallel_args = tqdm(
              sr_parallel_args, 
              desc=f'worker: {worker_id}, type: {subfolder}',
              #total=len(list(sr_parallel_args)),
          )
        
        results_chunked = \
            Parallel(
                n_jobs=nthreads, 
                backend=backend
            )(delayed(self._semrep_execute_)(*i) for i in sr_parallel_args)
        
            ## Merging end results ##
        
        #sr_output_filelist = list(temp_out_file_path.glob('*.out'))
        
        with open(temp_out_file_path, 'w') as fo:
            for out_filepath in sorted(sr_output_filelist):
                with open(out_filepath, 'r') as fi:
                    for line in fi:
                        fo.write(line)
        
        tqdm._instances.clear()
        
        if raw:
            #with open(temp_out_file_path) as f:
            #    parsed_sr_results = [
            #        line for line in f.read().split('\n') if line
            #    ]
            parsed_sr_results = temp_out_file_path
            if outputFilepath:
                copyfile(temp_out_file_path, outputFilepath)
                parsed_sr_results = outputFilepath
            print(f'Output is written to {parsed_sr_results}')
        else:
            parsed_sr_results = self.Parse_SemRep_results(temp_out_file_path)
            
        return parsed_sr_results
        
        
    
    def _semrep_execute_(
        self,
        input_filename:Path,
        output_filename:Path,
        parent_dir:Path,
        timeout_sec:int = 450,
    ) -> None:
        """Central SemRep function.
        
        Handles SemRep queries via subprocess python library.
        """
        iter = 0
        errorCode = 1
        sr_temp_path = parent_dir.joinpath(f'{input_filename.stem}')
        
        self._create_folder_(sr_temp_path)
        self._create_folder_(sr_temp_path.joinpath('semrepProcessed'))
        self._create_folder_(sr_temp_path.joinpath('logs'))
        
        time.sleep(1)
        
        while errorCode != 0:
            current_sr_output_path = sr_temp_path.joinpath(f'semrepProcessed/sr_out_{iter}.txt')
            
            current_log_path = sr_temp_path.joinpath(sr_temp_path.joinpath(f'logs/sr_log_{iter}.log'))
            #print("launching sr executable...")
            sr_process = subprocess.run(
                f'''timeout {timeout_sec} {self.sr_binary_path} {self.params} {input_filename} {current_sr_output_path} &> {current_log_path}''', 
                shell=True)
            errorCode = sr_process.returncode
            
            time.sleep(1)
            
            if errorCode != 0:
                bad_id = self._read_sr_log_(current_log_path)
                print("Bad id:", bad_id, 'Error code:', errorCode)

                self._remove_bad_record_(
                    input_filename, 
                    #c_input_filename, 
                    bad_id)
            iter += 1
        
        time.sleep(1)
        
        with open(output_filename, 'w') as fo:
            for sr_output_file in sorted(os.listdir(sr_temp_path.joinpath('semrepProcessed'))):
                with open(sr_temp_path.joinpath(f'semrepProcessed/{sr_output_file}'), 'r') as f:
                    #print('SEP FILE')
                    #print(f.read())
                    fo.write(f.read())
    
    def _read_sr_log_(
        self,
        log_path
    ) -> str:
        """
        Reads SemRep log file and detects where an error could occur.
        
        If "Output written to" phrase presented in the file, it means that no errors occurred at that run.
        If not, finds the latest successfull PMID and returns it.
        """
        
        with FileReadBackwards(log_path, encoding="utf-8") as f:
            # getting lines by lines starting from the last line up
            for line in f:
                if 'Output written to' in line:
                    return '0'
                else:
                    if "Processing" in line:
                        bad_id = line.split()[1] \
                            .split('.')[0]
                        return bad_id
            return 'unknown'
                    
    def _remove_bad_record_(
        self,
        inputFile:Path,
        bad_id:str,
    ) -> None:
        """Given a document and an ID of the last successful SemRep record, 
        it removes ALL records preceeding this record AND one more.
        This one more record presumably causes issues.
        
        Returns:
            None, operates on files.
        """

        buffer = []
        if bad_id == 'unknown':
            bad_id_found = True
        else:
            bad_id_found = False
        with open(inputFile, 'r') as f:
            for line in f:
                if bad_id in line:
                    bad_id_found = True
                if bad_id_found:
                    buffer.append(line)
                    
        with open(inputFile, 'w') as f:
            if len(buffer) > 2:
                for line in buffer[2:]:
                    f.write(line)
                print(f'Skipped record: {buffer[1]}')
        return 
    
    
    def Clean_temp_folder(self,
                          verbose:bool = True,
                          temp_path:Path = '',
                         ) -> None:
        if not temp_path:
            temp_path = self.temp_folder
        if temp_path.is_dir():
            try:
                res = subprocess.run(f'rm -r {temp_path}/*', shell = True)
                if verbose:
                    print(f'{temp_path}/ cleaned.')
            except Exception as e:
                print(e)
    
    def Killall_SemRep_instances(self):
        killall = subprocess.run(f'killall -9 semrep.v1.9.BINARY.Linux', shell=True)
        return killall
    
    def Parse_SemRep_results(
        self,
        sr_output_file:Path,
    ) -> dict:
        """Parses single semrep results file.
        
        Has 3 internal subfunctions to process different kind of stuff:
            - Sentences
            - Entities 
            - Relations
            
        Output:
            dict(dict) with the following structure:
            
            {sent_id:
                {
                    sent_text: <raw sent txt>,
                    terms: <[...]>,
                    relations: <[...]>,
                }
            }
        
        """
        
                ### INTERNAL FUNCTION DEFINITIONS ###
            
        def ExtractPMID(line):
            pr = line.split('|')
            return f'''s:{pr[1]}:{pr[4]}'''
        
        def ProcessSRSentence(
            line:str, 
            #sents:defaultdict(lambda: defaultdict(str)),
        ) -> None:
            '''
            Extracts sentences (their texts) from semrep output line.

            '''
            pr = line.split('|')

            pmid = sorted(pr[1].split(':'), 
                          key = lambda x: len(x), 
                          reverse = True
                         )[0]

            sentenceText = pr[-1].strip()

            sentID = pr[1]
            sentID = f'''s:{pr[1]}:{pr[4]}'''
                
            #if len(pr[1].split(':')) < 3 and 'titles' not in partN:
            #    sentID = f'''s:{sentID}:{pr[4]}'''
            #if 'titles' in partN:
            #    sentID = f'''s:{sentID}:0'''

            #sents[pmid][sentID] += f'{sentenceText} '
            
            #sents[sentID] += f'{sentenceText} '
            tempRes['sent_text'] += f'{sentenceText} '
            
            #sentsDict[sentID] += f'{sentenceText} '

            #return pmid, sentID, sentenceText

        def ProcessSREntity(
            line:str, 
            #ents:defaultdict(list),
        ) -> None:
            '''
            Process one SemRep entity and add it to the `ents` defaultdict.
            '''

            lettersPattern = re.compile('[a-zA-Z]')
            pr = line.split('|')
            try: 
                term = {
                    'CID': pr[6],
                    'pref_name': pr[7] if pr[7] else pr[11],
                    'extracted_text': pr[11],
                    'label': 'UMLS' if pr[6] else 'ENTITY',
                    'sem_types': pr[8].split(','),
                    'negated': True if pr[-4] == '1' else False,
                }

                #sentID = pr[1]
                #if len(pr[1].split(':')) < 3 and 'titles' not in partN:
                #    sentID = f'''s:{sentID}:{pr[4]}'''
                #if 'titles' in partN:
                #    sentID = f'''s:{sentID}:0'''
                    
                sentID = f'''s:{pr[1]}:{pr[4]}'''

                if re.match(lettersPattern, term['extracted_text']):
                    #ents[sentID].append(term)
                    tempRes['terms'].append(term)
            except:
                pass
            
        
        def ProcessSRRelation(
            line:str,
            #rels:defaultdict(list),
        ) -> None:
            '''
            Extracts relations from semrep output file line.
            The output looks like this:

            > abstrRelationsExtended['s:30863093:1:1']

            Output: ['C0027651|Neoplasms|tumors|PROCESS_OF|C0030705|Patients|patients',
                     'C0027651|Neoplasms|tumors|PROCESS_OF|C0030705|Patients|patients']

            '''
            pr = line.split('|')
            try:
                #term = f'{pr[8]}|{pr[22]}|{pr[28]}'
                #term = f'{pr[8]}|{pr[9]}|{pr[14]}|{pr[22]}|{pr[28]}|{pr[29]}|{pr[34]}'
                
                term = {
                    'subj_id': pr[8],
                    'subj_name': pr[9],
                    'subj_text': pr[14],
                    #'subj_sem_type': pr[10].split(','),
                    'subj_sem_type': pr[11],
                    'subj_negated': True if pr[17] == '1' else False,
                    
                    'verb': pr[22],
                    'verb_negated': True if pr[23] == '1' else False,
                    
                    'obj_id': pr[28],
                    'obj_name': pr[29],
                    'obj_text': pr[34],
                    'obj_sem_type': pr[31],
                    'obj_negated': True if pr[37] == '1' else False,                    
                }

                #sentID = pr[1]
                #if len(pr[1].split(':')) < 3 and 'titles' not in partN:
                #    sentID = f'''s:{sentID}:{pr[4]}'''
                #if 'titles' in partN:
                #    sentID = f'''s:{sentID}:0'''
                
                sentID = f'''s:{pr[1]}:{pr[4]}'''

                # we need only relations where there are 2 extracted UMLS terms
                # (sometimes we have relations w/o UMLS terms)
                if len(pr[8]) > 0 and len(pr[28]) > 0: 
                    #rels[sentID].append(term)
                    tempRes['relations'].append(term)
            except:
                pass

        ###################################

        #filenames = [f'{srOutputFolderPath}/{fn}' for fn in os.listdir(srOutputFolderPath) if '.txt' in fn]

        #for medline
        #workingFlist = [_ for _ in filenames if f'part-{partN}_' in _]

        #for cord
        #workingFlist = [_ for _ in filenames if f'{partN}' in _]

        #sents = defaultdict(lambda: defaultdict(str))
        #sentsDict = defaultdict(str)

        #ents = defaultdict(list)
        #rels = defaultdict(list)
        
        res = dict()
        
        with open(sr_output_file, 'r') as f:
            try:
                for line in f:
                    if len(line) > 1:
                        try:
                            pmid = ExtractPMID(line)
                        except:
                            pmid = ''
                        if pmid and pmid in res:
                            tempRes = res[pmid]
                        else:
                            tempRes = {
                                'sent_text': '',
                                'terms': [],
                                'relations': [],
                            }
                        if '|entity|' in line: # rule for entities
                            ProcessSREntity(line)
                        if line.count('|') == 40: # rule for relations
                            ProcessSRRelation(line)
                        if '|text|' in line: # rule for sentences
                            ProcessSRSentence(line)
                        if pmid:
                            res[pmid] = tempRes
            except Exception as e:
                print(e, sr_output_file)

        return res



## Function run from Agatha.construct pipeline
def sr_process_checkpoints(
    ckpt_records,
    nlm_soft_folder,
    sr_temp_folder,
    sr_replace_utf8_path,
    sr_params,
    sr_binary,
    show_progress=True,
):
    """Processes records from `<prefix>_documents` checkpoint with SemRep."""
    
    #print("Starting semrep with parameters:")
    #print(nlm_soft_folder)
    #print(sr_temp_folder)
    
    sr_handler = SemRepHandler(
        nlm_soft_path=nlm_soft_folder,
        temp_folder=sr_temp_folder,
        restart_mm_services=False,
        replace_utf8_path=sr_replace_utf8_path,
    )
    
    try:
        worker_id = socket.gethostname().split('.')[0]
    except:
        worker_id = 'unk worker'
    
    doc_titles_dict = dict()
    doc_text_dict = dict()
    
    abstr_metadata_dict = dict()
    
    for record in ckpt_records:
        title_text = []
        abstr_text = []
        
        # metadata without text
        record_metadata = record.copy()
        del record_metadata['text_data']
        abstr_metadata_dict[record['pmid']] = record_metadata
        
        # text data only
        for td in record['text_data']:
            if td['type'] == 'title':
                title_text.append(td['text'])
            else:
                abstr_text.append(td['text'])
                
        doc_titles_dict[record['pmid']] = ' '.join(title_text)
        doc_text_dict[record['pmid']] = ' '.join(abstr_text)
        
    # Running SemRep
    
    sr_handler.params = sr_params
    sr_handler.sr_binary_path = sr_binary
    
    sr_raw_results_titles = sr_handler.ProcessList_parallel(
        inputList=doc_titles_dict,
        chunkSize=100,
        backend='threading',
        subfolder='titles',
        show_progress=show_progress,
    )

    time.sleep(1)

    # Processing SemRep results
    
    #sr_raw_results_titles = dict()
    
    ## Titles
    sr_results_titles = dict()
    for sent_key in sr_raw_results_titles:
        pmid = sent_key.split(':')[1]
        sr_raw_results_titles[sent_key]['sent_type'] = 'title'
        if pmid in sr_results_titles:
            sr_results_titles[pmid]['sent_text'] += f"{sr_raw_results_titles[sent_key]['sent_text']}"
            sr_results_titles[pmid]['terms'] += sr_raw_results_titles[sent_key]['terms']
            sr_results_titles[pmid]['relations'] += sr_raw_results_titles[sent_key]['relations']
        else:
            sr_results_titles[pmid] = sr_raw_results_titles[sent_key]
           
    
    sr_raw_results_abstracts = sr_handler.ProcessList_parallel(
        inputList=doc_text_dict,
        chunkSize=50,
        backend='threading',
        subfolder='abstracts',
        show_progress=show_progress,
    )
    
    time.sleep(1)
    
    ## Abstracts
    sr_results_abstracts = defaultdict(list)
    for sent_key in sr_raw_results_abstracts:
        sr_raw_results_abstracts[sent_key]['sent_type'] = 'abstract:raw'
        sent_key_split = sent_key.split(':')
        pmid = sent_key_split[1]
        cur_sent_data = sr_raw_results_abstracts[sent_key]
        try:
            cur_sent_data['sent_idx'] = int(sent_key_split[-1])
            sr_results_abstracts[pmid].append(cur_sent_data)
        except Exception as e:
            print(f"Worker: {worker_id}")
            print(f"Failed to get sentence key. Sentence idx: {sent_key}")
            print(e)

    for pmid in sr_results_abstracts:
        sr_results_abstracts[pmid].sort(key=lambda x: x['sent_idx'])
        
    ## Merging both together to match `<prefix>_sentences` format
    sr_results_sentences = []
    
    for pmid in abstr_metadata_dict:
        pmid_sentences = []
        if pmid in sr_results_titles:
            sr_results_titles[pmid]['sent_idx'] = 0
            pmid_sentences.append(sr_results_titles[pmid])
        if pmid in sr_results_abstracts:
            pmid_sentences += sr_results_abstracts[pmid]
        sent_total = len(pmid_sentences)
        if sent_total > 0:
            for sent in pmid_sentences:
                sent.update(abstr_metadata_dict[pmid])
                sent['sent_total'] = sent_total
                sent['id'] = f"s:{sent['pmid']}:{sent['version']}:{sent['sent_idx']}"

            sr_results_sentences += pmid_sentences
    
    return sr_results_sentences
