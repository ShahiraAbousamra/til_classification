# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================

class Modes:
    TRAIN = 0;
    TEST = 1;

class ModelParams:
    def __init__(self):
        self.params = {};
        self.params['arch_name'] = 'basic';
        self.params['mode'] = Modes.TRAIN;
        self.params['n_classes'] = 2;
        self.params['n_channels'] = 1;
        self.params['train_out_path'] = '';
        self.params['test_out_path'] = '';
        self.params['validate_out_path'] = '';
