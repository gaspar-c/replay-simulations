import os
import time
from code_files.run_tests import run_test_group


if __name__ == '__main__':

    settings = {
        'save_plots': True,
        'max_cores': 2,
        'output_dir': os.getcwd() + '/outputs/',
        'group_name': 'test',
        'time_stamp': time.strftime("%Y%m%d_%H%M%S")
    }

    # network parameters
    param_arrays = {
        'conn_seed': [3],
        'n_p': [10000],
        'n_b': [2500],
        'n_p_asb': [800],
        'p_pp': [0.05, 0.07]
    }

    run_test_group(settings, param_arrays)
