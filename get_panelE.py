import os
import time
from my_code.run_tests import run_test_group
from my_code.plot_group_results import group_plot


if __name__ == '__main__':

    settings = {
        'save_plots': True,
        'max_cores': 31,
        'output_dir': os.getcwd() + '/outputs/',
        'group_name': 'panelE',
        'time_stamp': time.strftime("%Y%m%d_%H%M%S")
    }

    param_arrays = {
        'n_p': [50000],
        'n_b': [12500],
        'p_pp': [0.08],
        'conn_seed': [1] * 11 * 11 +
                     [2] * 11 * 11 +
                     [3] * 11 * 11 +
                     [4] * 11 * 11 +
                     [5] * 11 * 11,
        'gamma': ([3.10] * 11 +
                  [2.80] * 11 +
                  [2.50] * 11 +
                  [2.20] * 11 +
                  [1.90] * 11 +
                  [1.60] * 11 +
                  [1.30] * 11 +
                  [1.00] * 11 +
                  [0.70] * 11 +
                  [0.40] * 11 +
                  [0.10] * 11
                  ) * 5,
        'n_p_asb': [1500, 1400, 1300, 1200, 1100,
                    1000, 900, 800, 700, 600, 500] * 11 * 5,
    }

    run_test_group(settings, param_arrays)

    group_path = settings['output_dir'] + settings['group_name'] + '_' + settings['time_stamp']
    group_plot(group_path,
               'gamma', 'n_p_asb',
               r'Sequence Weight (pS)', r'Pattern Size $M$',
               title_=r'Connectivity $c$ = 8 \% \\ Network Size $N$ = ' + r'{:,}'.format(int(50000)),
               scale_x=50, skip_xlabel=3)
