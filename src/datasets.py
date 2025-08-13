import numpy as np

class ground_data:
    def __init__(self, filename, subsampling_rate, reference_CV = None):
        data_dir = np.load(filename)
        self.data = data_dir['data_all_atom']
        if reference_CV is not None:
            self.ref_vec = data_dir[reference_CV]
        self.subsampling_rate = subsampling_rate
    def return_data(self):
        self.data = self.data[::self.subsampling_rate]
        if hasattr(self, 'ref_vec'):
            self.ref_vec = self.ref_vec[::self.subsampling_rate]
        return self.data, self.ref_vec
    
