# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================

class AbstractDataProvider:
    def __init__(self, is_test, filepath_data, filepath_label, n_channels, n_classes, do_preprocess, do_augment, data_var_name=None, label_var_name=None, permute=False, repeat=True):
        self.is_test = is_test; 
        self.filepath_data = '';
        self.filepath_label = '';
        self.n_channels = n_channels;
        self.n_classes = n_classes;
        self.do_preprocess = do_preprocess;
        self.do_augment = do_augment;
        self.data_var_name = 'c488gb';
        self.label_var_name = label_var_name;
        self.do_permute = permute;
        self.do_repeat = repeat; # go over the data  multiple times like with training or only once like with test

        self.is_loaded = False;

    def load_data(self):
        self.data = None;
        self.label = None;
        self.last_fetched_indx = -1;
        self.permutation = None;
        self.data_count = 0;

    def reset(self, repermute=None):
        self.last_fetched_indx = -1;

    def get_next_one(self):
        pass;

    def get_next_n(self, n:int):
        pass;

    def preprocess(self, data_point, label):
        return data_point, label;

    def augment(self, data_point, label):
        return data_point, label;
