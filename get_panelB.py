import os
import time
from my_code.run_tests import run_test_group


if __name__ == '__main__':

    settings = {
        'save_plots': True,
        'max_cores': 2,
        'output_dir': os.getcwd() + '/outputs/',
        'group_name': 'panelB',
        'time_stamp': time.strftime("%Y%m%d_%H%M%S")
    }

    # network parameters
    param_arrays = {
        'conn_seed': [3],
        'n_p': [50000],
        'n_b': [12500],
        'n_p_asb': [1200],
        'p_pp': [0.07, 0.09]
    }

    run_test_group(settings, param_arrays)
