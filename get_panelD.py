import os
import time
from code_files.run_tests import run_test_group
from code_files.plot_group_results import group_plot


if __name__ == '__main__':

    settings = {
        'save_plots': True,
        'max_cores': 30,
        'output_dir': os.getcwd() + '/outputs/',
        'group_name': 'panelD',
        'time_stamp': time.strftime("%Y%m%d_%H%M%S")
    }

    param_arrays = {
        'p_pp': [0.08],
        'conn_seed': [1] * 9 * 9 +
                     [2] * 9 * 9 +
                     [3] * 9 * 9 +
                     [4] * 9 * 9 +
                     [5] * 9 * 9,
        'n_p': ([100000] * 9 +
                [90000] * 9 +
                [80000] * 9 +
                [70000] * 9 +
                [60000] * 9 +
                [50000] * 9 +
                [40000] * 9 +
                [30000] * 9 +
                [20000] * 9) * 5,
        'n_b': ([25000] * 9 +
                [22500] * 9 +
                [20000] * 9 +
                [17500] * 9 +
                [15000] * 9 +
                [12500] * 9 +
                [10000] * 9 +
                [7500] * 9 +
                [5000] * 9) * 5,
        'n_p_asb': [1500, 1400, 1300, 1200, 1100,
                    1000, 900, 800, 700] * 9 * 5,
    }

    run_test_group(settings, param_arrays)

    group_path = settings['output_dir'] + settings['group_name'] + '_' + settings['time_stamp']
    group_plot(group_path,
               'n_p', 'n_p_asb',
               r'Network Size $N$', r'Pattern Size $M$',
               title_=r'Connectivity $c$ = 8 \%\\Sequence weight: 50 pS',
               scale_x=0.001, xlabel_append='k')
