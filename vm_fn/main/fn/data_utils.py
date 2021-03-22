import numpy as np
import multiprocessing as mp
import json
from torch.utils.data import Dataset

from trlib.inputters.vector import vectorize
from trlib.inputters.utils import process_examples

class FileReader:
    def __init__(self, filename):
        self.fin = open(filename, "r")
        self.line_map = list() # Map from line index -> file position.
        self.line_map.append(0)
        while self.fin.readline():
            self.line_map.append(self.fin.tell())

    def get_line(self, index):
        self.fin.seek(self.line_map[index])
        return self.fin.readline()   

def read_lengths(tgt_filename, offset=0):
    num_exs = []
    with open(tgt_filename) as f:
        for line in f:
            num_exs.append(len(line.strip().split())+offset)
    return num_exs   
    
class NCSDataset(Dataset):
    def __init__(self, model, args, filenames):
        self.model = model
        self.filenames = filenames
        self.args = args
        self._lengths_code = read_lengths(filenames["src"], offset=0)
        self._lengths_sum = read_lengths(filenames["tgt"], offset=0)
        assert len(self._lengths_code) == len(self._lengths_sum)
        self.num_exs = len(self._lengths_code)
        self.fds = []
        self.locks = []
        self.num_fds = 0
        self.global_lock = mp.Lock()

    def __len__(self):
        return self.num_exs
    
    def get_fd(self):
        res = -1
        with self.global_lock:
            for i in range(self.num_fds):
                if not self.locks[i]:
                    res = i
                    break
            if res == -1:
                self.locks.append(False)
                self.fds.append({key:(FileReader(self.filenames[key]) \
                               if self.filenames[key] is not None \
                               else None) for key in self.filenames})
                res = self.num_fds
                self.num_fds += 1
            self.locks[res] = True
        return res

    def __getitem__(self, index):
        num_fd = self.get_fd()
        fd = self.fds[num_fd]
        ex = {}
        for key in fd:
            if fd[key] is not None:
                    f = fd[key]
                    line = f.get_line(index).strip()
                    if key == "rel_matrix":
                        line = json.loads(line)
                    ex[key] = line
            else:
                ex[key] = None
        with self.global_lock:
            self.locks[num_fd] = False
        ex_obj = process_examples(ex["src"],
                                   ex["src_tag"],
                                   None,
                                   ex["tgt"],
                                   ex["rel_matrix"],
                                   ex["root_paths"],
                                   ex["edges"],
                                   self.args.max_src_len,
                                   self.args.max_tgt_len,
                                   uncase=False,
                                   split_tokens=self.args.sum_over_subtokens)
        vector = vectorize(ex_obj, self.model)
        return vector
        
    def lengths(self):
        return [(self._lengths_code[i], self._lengths_sum[i])
                for i in range(self.num_exs)]