import os
import time
from my_code.run_tests import run_test_group
from my_code.plot_group_results import group_plot


if __name__ == '__main__':

    settings = {
        'save_plots': True,
        'max_cores': 30,
        'output_dir': os.getcwd() + '/outputs/',
        'group_name': 'panelC',
        'time_stamp': time.strftime("%Y%m%d_%H%M%S")
    }

    # network parameters
    param_arrays = {
        'n_p': [50000],
        'n_b': [12500],
        'conn_seed': [1] * 11 * 11 +
                     [2] * 11 * 11 +
                     [3] * 11 * 11 +
                     [4] * 11 * 11 +
                     [5] * 11 * 11,
        'n_p_asb': (
                    [1500] * 11 +
                    [1400] * 11 +
                    [1300] * 11 +
                    [1200] * 11 +
                    [1100] * 11 +
                    [1000] * 11 +
                    [900] * 11 +
                    [800] * 11 +
                    [700] * 11 +
                    [600] * 11 +
                    [500] * 11) * 5,
        'p_pp': [0.14, 0.13, 0.12, 0.11, 0.10, 0.09,
                 0.08, 0.07, 0.06, 0.05, 0.04] * 11 * 5,
    }

    run_test_group(settings, param_arrays)

    group_path = settings['output_dir'] + settings['group_name'] + '_' + settings['time_stamp']
    group_plot(group_path,
               'p_pp', 'n_p_asb',
               r'Connectivity $c$ (\%)', r'Pattern Size $M$',
               title_=r'Network Size $N = $ %s' % ('{:,}'.format(int(50000))),
               scale_x=100, fit_inv=True)
