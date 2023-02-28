from brian2 import *
import os
import multiprocessing
import time
import random
import matplotlib.pyplot as plt
from my_code import network as net, parameters as param, tests, plots
from my_code.aux_functions import get_logger, pprint, s_to_hh_mm_ss


def run_test_group(settings, param_arrays):
    """
    runs a group of tests in parallel

    Args:
        settings: settings for the test group
        param_arrays: arrays with parameters for each test in the group

    """

    """ CREATE TEST GROUP OUTPUT FOLDER """
    group_log = settings['output_dir'] + settings['group_name'] + '_' + settings['time_stamp'] + '/0_group_log.log'
    os.mkdir(settings['output_dir'] + settings['group_name'] + '_' + settings['time_stamp'])

    """ COUNT NUMBER OF TESTS TO RUN """
    n_tests = 0
    for par in param_arrays:
        if len(param_arrays[par]) > n_tests:
            n_tests = len(param_arrays[par])

    pprint('Running %d tests ...' % n_tests, group_log)

    """ COUNT HOW MANY CPU CORES TO USE """
    cpu_cores = multiprocessing.cpu_count()
    max_cores = min([settings['max_cores'], cpu_cores])
    n_cores = min([n_tests, max_cores])

    """ CREATE TABLE WITH PARAM ARRAY IN GROUP LOG FILE """
    lines = [''] * (n_tests + 1)

    # header
    lines[0] = 'test#'
    for par in param_arrays:
        lines[0] += ' \t\t ' + par

    # param array
    for i in range(n_tests):
        lines[i + 1] = '%s' % (i + 1)
        for par in param_arrays:
            if i < len(param_arrays[par]):
                val_p = param_arrays[par][i]
            else:
                val_p = param_arrays[par][-1]
            lines[i + 1] += ' \t\t %s' % (val_p,)

    # print to file
    for i in range(len(lines)):
        pprint(lines[i], group_log)

    """ CREATE TEST RESULTS FILE """

    results_file = open(settings['output_dir'] + settings['group_name'] + '_' + settings['time_stamp']
                        + '/0_group_results.txt', 'w')

    results_header = 'test# \t '
    for par in param_arrays:
        results_header += par + ' \t '
    results_header += 'stim# \t replay'
    results_file.write(results_header + '\n')
    results_file.close()

    """ RUN TESTS """
    pprint('Running %d tests in %d/%d CPUs' % (n_tests, n_cores, cpu_cores), group_log)
    start_time = time.time()
    if n_cores == 1:
        for i in range(n_tests):
            test_idx = i + 1
            settings_single = choose_from_param_arrays(settings, param_arrays, test_idx, n_cores)
            run_single_test(settings_single)

    # parse n_tests by n_cores
    elif n_cores > 1:
        test_idx = 1
        n_tests_left = n_tests
        n_steps = (n_tests // n_cores) + 1
        for _ in range(n_steps):
            if n_tests_left > 0:
                n_cores_step = min(n_cores, n_tests_left)

                pprint('\t %s: running tests %d-%d in %d CPUs' % (time.strftime("%H:%M:%S"),
                                                                  test_idx, test_idx + n_cores_step - 1,
                                                                  n_cores_step), group_log)

                start_step_time = time.time()

                settings_array = [None] * n_cores_step
                for i in range(n_cores_step):
                    settings_array[i] = choose_from_param_arrays(settings, param_arrays, test_idx, n_cores_step)
                    test_idx += 1

                par = multiprocessing.Pool(n_cores_step)
                par.map(run_single_test, settings_array)
                n_tests_left -= n_cores_step

                end_step_time = time.time() - start_step_time
                pprint('\t\t %s: finished step in %s. %d tests left...' %
                       (time.strftime("%H:%M:%S"), s_to_hh_mm_ss(end_step_time), n_tests_left), group_log)

    else:
        raise ValueError('Number of cores must be >= 1!')

    end_time = time.time() - start_time
    pprint('Finished test group in %s' % (s_to_hh_mm_ss(end_time)), group_log)


def choose_from_param_arrays(settings, param_arrays, test_idx, n_cores):
    """
    create settings dictionary for a single test

    Args:
        settings: test group settings
        param_arrays: parameter arrays for all tests in group
        test_idx: index of the single test
        n_cores: number of cpu cores

    Returns:
        test_settings: settings dictionary for single test

    """
    # copy group settings to single test settings
    test_settings = settings.copy()

    # test index and cpu in which test will run
    test_settings['test_idx'] = test_idx
    test_settings['core_idx'] = (test_idx % n_cores) + 1

    # select test parameters from the param arrays for all tests
    select_params = {}
    for par in param_arrays:
        param_array = param_arrays[par]
        if len(param_array) > (test_idx - 1):
            select_params[par] = param_array[test_idx - 1]
        else:
            select_params[par] = param_array[-1]
    test_settings['param_array'] = select_params

    return test_settings


def run_single_test(sett_dict):
    """
    run a single test from the group tests
    (this function will run in parallel)

    Args:
        sett_dict: dictionary with settings for the single test to be run

    """

    # log command line
    log_name = get_logger(sett_dict)

    # get default parameters
    net_params = param.get_dft_net_params()
    test_params = param.get_dft_test_params()
    plot_params = param.get_dft_plot_params()

    # load array with parameters for this test
    if 'param_array' in sett_dict:
        pprint('Network parameters:', log_name)
        param.load_specified_params(net_params, sett_dict['param_array'], logfile=log_name)

    # scale synaptic weight:
    gamma_val = net_params['gamma'].get_param()
    net_params['g_pp'] = param.Parameter(net_params['g_pp'].get_param() * gamma_val / nS, nS)
    net_params['g_bp'] = param.Parameter(net_params['g_bp'].get_param() * gamma_val / nS, nS)
    net_params['g_pb'] = param.Parameter(net_params['g_pb'].get_param() * gamma_val / nS, nS)
    net_params['g_bb'] = param.Parameter(net_params['g_bb'].get_param() * gamma_val / nS, nS)
    # log-normal distribution
    net_params['g_pp_median'] = param.Parameter(net_params['g_pp'].get_param() * 0.02 / nS, nS)

    # test events
    events = {}

    # turn on STDP
    stdp_on_time = test_params['pb_stdp_start'].get_param()
    events['stdp_on'] = tests.ChangeAttribute(time_=stdp_on_time,
                                              target='syn_pb',
                                              attribute='eta_switch',
                                              value=1)

    # turn off STDP
    stdp_off_time = stdp_on_time + test_params['pb_stdp_dur'].get_param()
    events['stdp_off'] = tests.ChangeAttribute(time_=stdp_off_time,
                                               target='syn_pb',
                                               attribute='eta_switch',
                                               value=0)

    # stimulate first assembly to trigger replay
    events['stim_1'] = tests.Stimulus(target='p', onset=stdp_off_time + 0.1 * second, length=5 * ms,
                                      strength=150 * pA, asb=1, prop=0.5)
    events['stim_2'] = tests.Stimulus(target='p', onset=stdp_off_time + 1.1 * second, length=5 * ms,
                                      strength=150 * pA, asb=1, prop=0.5)
    events['stim_3'] = tests.Stimulus(target='p', onset=stdp_off_time + 2.1 * second, length=5 * ms,
                                      strength=150 * pA, asb=1, prop=0.5)
    events['stim_4'] = tests.Stimulus(target='p', onset=stdp_off_time + 3.1 * second, length=5 * ms,
                                      strength=150 * pA, asb=1, prop=0.5)
    events['stim_5'] = tests.Stimulus(target='p', onset=stdp_off_time + 4.1 * second, length=5 * ms,
                                      strength=150 * pA, asb=1, prop=0.5)

    # get test limits
    event_keys = list(events.keys())
    test_lims = [[stdp_off_time - 400 * ms, stdp_off_time]]  # final 400 ms of STDP stabilisation
    last_stim_time = 0 * second
    for i in range(len(event_keys)):
        if isinstance(events[event_keys[i]], tests.Stimulus):
            stim_onset = events[event_keys[i]].onset
            if stim_onset > last_stim_time:
                last_stim_time = stim_onset

            lim_start = stim_onset - 200 * ms
            lim_stop = lim_start + 400 * ms
            test_lims += [[lim_start, lim_stop]]  # [-200ms, +200ms] around stimulation
    test_params['sim_time'] = param.Parameter((last_stim_time / second) + 0.2, second)
    test_lims += [[0 * ms, test_params['sim_time'].get_param()],  # whole simulation range
                  [0 * ms, 400 * ms]]  # first 400 ms

    """ BUILD BRIAN NETWORK """

    # initialise python seed:
    test_seed = test_params['test_seed'].get_param()
    random.seed(test_seed)

    sim_dt = test_params['sim_dt'].get_param()
    defaultclock.dt = sim_dt

    # create network:
    built_network = net.build_network(net_params, logfile=log_name)

    # add brian monitors for recording
    net.record_network(built_network, net_params, test_params)

    # initialise network with the specified initial conditions
    net.init_network(built_network, net_params, test_params)

    """ RUN TEST """

    pprint('\n================ Starting Test =================\n', log_name)

    # run actual brian simulation:
    start_time = time.time()
    tests.run_simulation(built_network, test_params, net_params, events, logfile=log_name)
    test_dur = time.time() - start_time
    pprint("Finished simulation in %s" % s_to_hh_mm_ss(test_dur), log_name)

    pprint('\n================ Finished Test =================\n', log_name)

    """ GET TEST RESULTS """

    for i in range(len(test_lims)):
        lims = test_lims[i]

        # prepare brian monitors for plotting and test calculations:
        ready_monitors, test_data = tests.prepare_test(built_network, test_params, lims[0], lims[1])

        # print test results:
        pprint('\n========== TEST RESULTS %s ==================' % lims, log_name)
        tests.print_test_results(ready_monitors, test_data, net_params, test_params,
                                 events, sett_dict, logfile=log_name)

        # save plots:
        if sett_dict['save_plots']:
            if lims is None:
                pprint('Creating plot ...', log_name)
            else:
                pprint('Creating plot for [%.2f-%.2f]s ...' % (lims[0] / second, lims[1] / second), log_name)

            start_plot_time = time.time()
            fig_to_save = plots.create_test_figure(net_params, plot_params, test_data, ready_monitors, events)
            fig_name = sett_dict['output_dir'] + sett_dict['group_name'] + '_' + \
                sett_dict['time_stamp'] + '/' + str(sett_dict['test_idx'])
            if lims is not None:
                fig_name += '_[%.2f-%.2f]s' % (lims[0] / second, lims[1] / second)
            fig_name += '.png'
            fig_to_save.savefig(fig_name, dpi=300)
            plt.close(fig_to_save)

            end_plot_time = time.time() - start_plot_time
            pprint('Created plot for [%.2f-%.2f]s in %.2f seconds' %
                   (lims[0] / second, lims[1] / second, end_plot_time), log_name)

    pprint('\n=========== Used Network Parameters ============\n', log_name)
    for par in net_params:
        if net_params[par].used:
            pprint('%s = %s' % (par, net_params[par].get_param()), log_name)

    pprint('\n============= Used Test Parameters =============\n', log_name)
    for par in test_params:
        if test_params[par].used:
            pprint('%s = %s' % (par, test_params[par].get_param()), log_name)

    pprint('\n========== Finished Printing Results ===========\n', log_name)
