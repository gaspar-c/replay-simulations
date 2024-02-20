from brian2 import *
from code_files.detect_peaks import detect_peaks
from code_files.aux_functions import calc_rate_monitor, check_brian_monitor,\
    calc_network_frequency, trim_brian_monitor, calc_unit_rates, \
    calc_isi_cv, pprint
import numpy as np


def prepare_test(network, test_params, calc_start, calc_stop):
    """
    prepare brian monitors for test plots and calculations

    Args:
        network: brian network
        test_params: test parameters
        calc_start: time when calculations start
        calc_stop: time when calculations stop

    Returns:
        ready_monitors: monitor data ready for plotting
        test_data: calculated test data

    """

    sim_dt = test_params['sim_dt'].get_param()               # simulation step time
    max_freq = test_params['max_net_freq'].get_param() / Hz  # maximum frequency for power spectral densities

    """ SELECT TIME LIMITS FOR PLOTTING AND CALCULATIONS """

    ready_monitors = {}
    test_data = {}
    test_data['calc_start'] = calc_start
    test_data['calc_stop'] = calc_stop

    """ SPIKE MONITOR OPERATIONS """

    # create array of all possible spike monitors:
    spm_array = ['spm_p', 'spm_b', 'spm_p_out']
    asb_idx = 1
    while asb_idx < 1000:
        if check_brian_monitor(network, 'spm_p_asb_' + str(asb_idx), 'i'):
            spm_array.append('spm_p_asb_' + str(asb_idx))
            asb_idx += 1
        else:
            break

    # trim all existing spike monitors:
    for spm in spm_array:
        if check_brian_monitor(network, spm, 'i'):
            # prepare spike monitor for plotting:
            ready_monitors[spm] = trim_brian_monitor(network[spm], network[spm].i, 1, calc_start, calc_stop)

            pop_name = spm[4:]

            # calculate unit firing rates:
            mean_rate, std_rate = calc_unit_rates(network[spm], calc_start, calc_stop)
            test_data['mean_rate_' + pop_name] = mean_rate
            test_data['std_rate_' + pop_name] = std_rate

            # calculate inter-spike-interval coefficient of variation:
            mean_cv, std_cv, check_cv = calc_isi_cv(network[spm], calc_start, calc_stop)
            test_data['mean_cv_' + pop_name] = mean_cv
            test_data['std_cv_' + pop_name] = std_cv
            test_data['check_cv_' + pop_name] = check_cv

    """ RATE MONITOR OPERATIONS """

    # smoothing window for population rate calculation:
    smooth_window = test_params['smooth_window'].get_param()

    for spm in spm_array:
        if check_brian_monitor(network, spm, 'i'):
            rtm = 'rt' + spm[2:]

            # prepare rate monitor for plotting with a smooth gaussian window:
            ready_monitors[rtm] = calc_rate_monitor(network[spm], calc_start, calc_stop, sim_dt, smooth_window)

            # calculate power spectral density of population rate:
            pop_name = rtm[4:]

            # trim raw population rate monitor:
            _, raw_rtm = calc_rate_monitor(network[spm], calc_start, calc_stop, sim_dt)

            # get power spectrum of population rate in the [0, max_freq] range and fit it to a lorentzian:
            pop_freq, pop_psd, net_freq, q_factor = calc_network_frequency(raw_rtm, calc_stop - calc_start,
                                                                           sim_dt / second, max_freq, fit=True)
            ready_monitors['psd_' + pop_name] = pop_freq, pop_psd
            test_data['net_freq_' + pop_name] = net_freq
            test_data['q_factor_' + pop_name] = q_factor

    """ STATE MONITOR OPERATIONS """

    # prepare adaptation plots:
    if check_brian_monitor(network, 'stm_p_adapt', 'w'):
        ready_monitors['stm_p_adapt'] = trim_brian_monitor(network['stm_p_adapt'],
                                                           np.mean(network['stm_p_adapt'].w, axis=0), pA,
                                                           calc_start, calc_stop)

    if check_brian_monitor(network, 'stm_stdp_pb_g', 'g_pb'):
        ready_monitors['stm_stdp_pb_g'] = network['stm_stdp_pb_g']

    if check_brian_monitor(network, 'stm_stdp_pb_eta', 'eta'):
        ready_monitors['stm_stdp_pb_eta'] = trim_brian_monitor(network['stm_stdp_pb_eta'],
                                                               network['stm_stdp_pb_eta'].eta[0], 1,
                                                               calc_start, calc_stop)

    return ready_monitors, test_data


class ChangeAttribute:
    """
    object used to change an attribute in the brian network while a simulation runs
    """
    def __init__(self, time_, target, attribute, value):
        self.time = time_
        self.target = target
        self.attribute = attribute
        self.value = value


class Stimulus:
    """ object used to stimulate neurons in the brian network while the simulation runs """
    def __init__(self, target, onset, length, strength,
                 asb=0, prop=1.0, rand_=False, shade=True):
        self.target = target
        self.onset = onset
        self.length = length
        self.strength = strength
        self.asb = asb
        self.prop = prop
        self.rand = rand_
        self.shade = shade
        self.applied = None
        self.checked_replay = False

    def store_applied(self, curr):
        self.applied = curr

    def get_applied(self, size_):
        if self.applied is None:
            return np.zeros(size_)
        else:
            return self.applied


def stim_cells(network, stimulus, net_params, test_params, off=False):
    """
    stimulate neurons in the network

    Args:
        network: brian network object
        stimulus: simulus object
        net_params: network parameters
        test_params: test parameters
        off: [True/False], if on, starts the stimulus, if off, ends it

    Returns:

    """
    pop_str = stimulus.target
    pop = network['pop_' + pop_str]
    rand_stim = stimulus.rand
    stim_amp = stimulus.strength
    stim_frac = stimulus.prop
    stim_asb = stimulus.asb

    if stim_asb > 0:
        n_i_asb = int(net_params['n_' + pop_str + '_asb'].get_param())
        pop_size = n_i_asb
        stim_idx0 = (stim_asb - 1) * n_i_asb
    else:
        pop_size = pop.N
        stim_idx0 = 0

    # +1 if 'on', -1 if 'off'
    switch = -(int(off) * 2 - 1)

    test_seed = int(test_params['test_seed'].get_param())

    # numpy seed must be initialized here to make sure that the random stimuli are turned off
    np.random.seed(test_seed)

    if stim_frac == 1.0:
        stim_idx = np.arange(stim_idx0, stim_idx0 + pop_size)
        stim_to_apply = switch * stim_amp * (1 - rand_stim * (1 - np.random.rand(pop_size)))

        present_stim = stimulus.get_applied(len(stim_idx))
        stimulus.store_applied(stim_to_apply)

        pop[stim_idx].curr_stim = present_stim + stim_to_apply

    else:
        stim_idx = stim_idx0 + np.random.default_rng(test_seed).choice(pop_size,
                                                                       size=int(stim_frac * pop_size),
                                                                       replace=False)

        stim_to_apply = switch * stim_amp * (1 - rand_stim * np.random.rand(len(stim_idx)))

        present_stim = stimulus.get_applied(len(stim_idx))
        stimulus.store_applied(stim_to_apply)

        i = 0
        for idx in stim_idx:
            pop[idx].curr_stim = present_stim[i] + stim_to_apply[i]
            i += 1


def run_simulation(network, test_params, net_params, events, logfile=None):
    """
    run network simulation

    Args:
        network: brian network object where test will be
        test_params: test parameters
        net_params: network parameters used to build network
        events: events to occur during the test
        logfile: log file

    """

    sim_time = test_params['sim_time'].get_param()

    # run simulation events
    if len(events) > 0:

        # get events and sort them
        event_keys = list(events.keys())
        event_names = []
        event_times = []
        event_types = []
        for key_ in event_keys:
            if isinstance(events[key_], Stimulus):
                event_names += [key_ + '_on']
                event_times += [events[key_].onset]
                event_types += ['stimulus_on']

                event_names += [key_ + '_off']
                event_times += [events[key_].onset + events[key_].length]
                event_types += ['stimulus_off']

            elif isinstance(events[key_], ChangeAttribute):
                event_names += [key_]
                event_times += [events[key_].time]
                event_types += ['change_attribute']

        event_times_norm = [t / second for t in event_times]
        event_order = np.argsort(event_times_norm)
        ordered_event_names = [event_names[i] for i in event_order]
        ordered_event_times = [event_times[i] for i in event_order]
        ordered_event_types = [event_types[i] for i in event_order]

        # run events in order
        for i in range(len(ordered_event_names)):
            time_to_next_event = ordered_event_times[i] - network.t
            if network.t + time_to_next_event > sim_time:
                break

            if time_to_next_event / second > 0:
                network.run(time_to_next_event, report='text', report_period=60 * second)

            event_name = ordered_event_names[i]
            if ordered_event_types[i] == 'stimulus_on':
                stim_cells(network, events[event_name[:-3]], net_params, test_params)

            elif ordered_event_types[i] == 'stimulus_off':
                stim_cells(network, events[event_name[:-4]], net_params, test_params, off=True)

            elif ordered_event_types[i] == 'change_attribute':
                if hasattr(network[events[event_name].target], events[event_name].attribute):
                    setattr(network[events[event_name].target], events[event_name].attribute, events[event_name].value)
                    pprint('%.4f s: changed %s.%s to %s' %
                           (network.t / second,
                            events[event_name].target,
                            events[event_name].attribute,
                            events[event_name].value), logfile)
                else:
                    pprint('ERROR: Failed to change attribute %s' % event_name, logfile)

    # run network for the remaining test time
    time_left = sim_time - network.t
    network.run(time_left, report='text', report_period=60 * second)


def print_test_results(monitors, test_data, net_params, test_params, events, sett_dict, logfile=None):
    """
    print test results

    Args:
        monitors: prepared monitors with simulation recordings
        test_data: calculated test data
        net_params: network parameters
        test_params: test parameters
        events: events that occur during the test
        sett_dict: settings for the test being run
        logfile: log file

    """

    sim_dt = test_params['sim_dt'].get_param()
    calc_start = test_data['calc_start']
    calc_stop = test_data['calc_stop']

    # quality of replay
    stim_keys = []
    for key_ in list(events.keys()):
        if isinstance(events[key_], Stimulus):
            stim_keys += [key_]

    stim_start = []
    stim_stop = []
    stim_unchecked = []
    for key_ in stim_keys:
        stim_start += [events[key_].onset]
        stim_stop += [events[key_].onset + events[key_].length]
        stim_unchecked += [not events[key_].checked_replay]

    # select stimuli that are unchecked and within calc_range:
    stim_start = np.array([t / second for t in stim_start]) * second
    stim_stop = np.array([t / second for t in stim_stop]) * second
    stim_unchecked = np.array(stim_unchecked)
    stim_check = (stim_start > calc_start) * (stim_start < calc_stop) *\
                 (stim_stop > calc_start) * (stim_stop < calc_stop) * stim_unchecked

    if np.sum(stim_check) == 1:
        stim_idx = np.argwhere(stim_check)[0][0]
        events[stim_keys[stim_idx]].checked_replay = True

        spm_asb_1_time, spm_p_asb_1_i = monitors['spm_p_asb_1']
        asb1_spiked, asb1_counts = np.unique(
            spm_p_asb_1_i[(spm_asb_1_time >= stim_start[stim_idx] / second) &
                          (spm_asb_1_time <= stim_stop[stim_idx] / second)],
            return_counts=True)
        n_p_asb = int(net_params['n_p_asb'].get_param())
        pprint('%d / %d neurons spiked (%.1f%%)' % (len(asb1_spiked), n_p_asb, 100*len(asb1_spiked)/n_p_asb), logfile)
        pprint('\t %d more than once' % np.sum(asb1_counts >= 2), logfile)

        n_asb = int(net_params['n_asb'].get_param())

        check_replay = True

        prev_peak = 0 * second
        replay_delay_min = 1 * ms
        replay_delay_max = 20 * ms

        for i in range(n_asb):
            rtm_p_asb_mon = monitors['rtm_p_asb_' + str(i + 1)]
            rtm_p_asb_t = rtm_p_asb_mon[0] * second
            rtm_p_asb_r = rtm_p_asb_mon[1] * Hz

            peak_idx = detect_peaks(rtm_p_asb_r, mph=60*Hz, mpd=int(1*ms / sim_dt))

            peak_times = rtm_p_asb_t[peak_idx]
            peak_heights = rtm_p_asb_r[peak_idx]

            pprint('asb %d:' % (i + 1), logfile)

            pprint('\t all peaks: %s : %s' % (peak_times, peak_heights), logfile)

            # detection range for 1st asb:
            if i == 0:
                detect_min = stim_start[stim_idx]
                detect_max = stim_stop[stim_idx] + replay_delay_max
            # detection range for subsequent assemblies:
            else:
                detect_min = prev_peak + replay_delay_min
                detect_max = prev_peak + replay_delay_max

            # find which of the detected peaks corresponds to evoked replay:
            evk_peak_idx = (peak_times >= detect_min) & \
                           (peak_times <= detect_max)

            pprint('\t replay peak: %s : %s' % (peak_times[evk_peak_idx], peak_heights[evk_peak_idx]), logfile)

            # only one 'evoked peak' should be detected within the detection range
            if np.sum(evk_peak_idx) == 0:
                pprint('Replay FAILED on asb %d: no peaks detected within detection range' % (i + 1), logfile)
                check_replay = False
                prev_peak = 0 * second

            elif np.sum(evk_peak_idx) == 1:
                pprint('\t delay: %s' % (peak_times[evk_peak_idx] - prev_peak), logfile)
                prev_peak = peak_times[evk_peak_idx]
                if peak_heights[evk_peak_idx] > 360 * Hz:
                    pprint('Replay FAILED on asb %d: detected peak is higher than threshold' % (i + 1), logfile)
                    check_replay = False
            else:
                pprint('Replay FAILED on asb %d: too many peaks detected' % (i + 1), logfile)
                check_replay = False
                prev_peak = 0 * second

        # check dummy group remains inactive between stim and last asb peak:
        if check_replay:
            detect_min = stim_start[stim_idx]
            detect_max = prev_peak
            rtm_p_out_mon = monitors['rtm_p_out']
            rtm_p_out_t = rtm_p_out_mon[0] * second
            rtm_p_out_r = rtm_p_out_mon[1] * Hz

            rtm_p_out_check = rtm_p_out_r[(rtm_p_out_t >= detect_min) & (rtm_p_out_t <= detect_max)]
            if (rtm_p_out_check > 30*Hz).any():
                check_replay = False
                pprint('Replay FAILED: dummy group exceeded threshold activity', logfile)
            else:
                pprint('Dummy group remained below threshold activity', logfile)

        """ SAVE RESULTS in TEST_RESULTS FILE """
        results_file = open(sett_dict['output_dir'] + sett_dict['group_name'] + '_' +
                            sett_dict['time_stamp'] + '/0_group_results.txt', 'r')
        array_param_names = results_file.readline()
        results_file.close()

        param_vals = '%d \t ' % sett_dict['test_idx']
        for param_name in array_param_names.split()[1:]:
            if (param_name == 'stim#') or (param_name == 'replay'):
                pass
            elif param_name in net_params:
                param_vals += net_params[param_name].get_str() + ' \t '
            else:
                pprint('ERROR: param %s not recognized' % param_name, logfile)
                param_vals += 'N/A \t '

        param_vals += str(stim_keys[stim_idx]) + ' \t '

        if check_replay:
            pprint('Replay SUCCEEDED!', logfile)
            param_vals += 'True'
        else:
            pprint('Replay FAILED!', logfile)
            param_vals += 'False'

        results_file = open(sett_dict['output_dir'] + sett_dict['group_name'] + '_' +
                            sett_dict['time_stamp'] + '/0_group_results.txt', 'a')
        results_file.write(param_vals + '\n')
        results_file.close()

    pprint('\n============== UNIT FIRING RATES ===============', logfile)

    # create array of all possible populations for which mean unit rate was calculated:
    mean_rate_array = ['mean_rate_p', 'mean_rate_b', 'mean_rate_p_out']
    asb_idx = 1
    while asb_idx < 1000:
        if 'mean_rate_p_asb_' + str(asb_idx) in test_data:
            mean_rate_array.append('mean_rate_p_asb_' + str(asb_idx))
            asb_idx += 1
        else:
            break

    # print unit firing rates:
    for mean_rate_str in mean_rate_array:
        if mean_rate_str in test_data:
            pop_name = mean_rate_str[10:]
            mean_rate = test_data[mean_rate_str]
            std_rate = test_data['std_rate_' + pop_name]
            pprint('%s firing rate (%.2f +/- %.2f) Hz' % (pop_name.upper(), mean_rate, std_rate), logfile)

    pprint('\n========== REGULARITY OF UNIT FIRING ===========', logfile)

    # create array of all possible populations for which isi cv was calculated:
    mean_cv_array = ['mean_cv_p', 'mean_cv_b', 'mean_cv_p_out']
    asb_idx = 1
    while asb_idx < 1000:
        if 'mean_cv_p_asb_' + str(asb_idx) in test_data:
            mean_cv_array.append('mean_cv_p_asb_' + str(asb_idx))
            asb_idx += 1
        else:
            break

    # print ISI CVs:
    for mean_cv_str in mean_cv_array:
        if mean_cv_str in test_data:
            pop_name = mean_cv_str[8:]
            mean_cv = test_data[mean_cv_str]
            std_cv = test_data['std_cv_' + pop_name]
            check_cv = test_data['check_cv_' + pop_name]
            if check_cv:
                pprint('%s ISI CV = %.2f +/- %.2f' % (pop_name.upper(), mean_cv, std_cv), logfile)
            else:
                pprint('%s ISI CV could not be calculated!' % pop_name.upper(), logfile)

    pprint('\n================ SYNCHRONICITY =================', logfile)

    # create array of all possible populations for which Q-factor was calculated:
    q_factor_array = ['q_factor_p', 'q_factor_b', 'q_factor_p_out']
    asb_idx = 1
    while asb_idx < 1000:
        if 'q_factor_p_asb_' + str(asb_idx) in test_data:
            q_factor_array.append('q_factor_p_asb_' + str(asb_idx))
            asb_idx += 1
        else:
            break

    # print synchronicity and Q-factor:
    q_factor_thres = test_params['q_factor_thres'].get_param()
    for q_factor_str in q_factor_array:
        if q_factor_str in test_data:
            pop_name = q_factor_str[9:]
            q_factor = test_data[q_factor_str]
            net_freq = test_data['net_freq_' + pop_name]
            if q_factor is None:
                pprint('%s is Asynchronous: Q-factor could not be calculated' % pop_name.upper(), logfile)
            elif q_factor <= q_factor_thres:
                pprint('%s is Asynchronous: Q-factor = %s (<= %.2f)' %
                       (pop_name.upper(), '{:.2e}'.format(q_factor), q_factor_thres), logfile)
            elif q_factor > q_factor_thres:
                pprint('%s network freq. %.2f Hz; Q-factor = %.2f (> %.2f)' %
                       (pop_name.upper(), net_freq, q_factor, q_factor_thres), logfile)

    if 'net_freq_lowpass_lfp' in test_data:
        pprint('Lowpass freq: %.2f Hz (q-factor %.2f)' %
               (test_data['net_freq_lowpass_lfp'], test_data['q_factor_lowpass_lfp']), logfile)

    if 'net_freq_bandpass_lfp' in test_data:
        pprint('Bandpass freq: %.2f Hz (q-factor %.2f)' %
               (test_data['net_freq_bandpass_lfp'], test_data['q_factor_bandpass_lfp']), logfile)
