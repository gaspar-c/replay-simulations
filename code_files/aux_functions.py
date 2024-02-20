from brian2 import *
import scipy.optimize
import numpy as np
import time


class ConnectivityMatrix:
    def __init__(self, conn, n_pre, n_post):
        """
        create connectivity matrix object
        """

        self.conn = conn            # probability of connection
        self.n_pre = int(n_pre)     # number of pre-synaptic cells
        self.n_post = int(n_post)   # number of post-synaptic cells

        self.seed = None      # pseudo-random seed
        self.pre_idx = None   # indices of pre-synaptic neurons
        self.post_idx = None  # indices of post-synaptic neurons
        self.n_syn = 0        # number of synapses

    def create_conn(self, seed_, logfile=None):
        """
        create pseudo-random connections
        """

        self.seed = seed_
        np.random.seed(self.seed)

        # number of possible connections
        total_size = self.n_pre * self.n_post

        # initialise array to store synapses with 1.10x the expected size
        synapses = np.zeros(int(total_size * self.conn * 1.1), dtype=np.uint64)

        # make max_size smaller if computer runs out of space for calculation
        # the smaller the number, the longer the calculation will take
        max_size = int(2 * 1e9)

        total_start_time = time.time()
        count_step = 0
        count_syn = 0
        size_left = total_size
        if total_size > max_size:
            time_spent = 0

            pprint('\t calculating ca. %s out of %s (%s x %s) possible synapses...' %
                   ('{:,}'.format(int(total_size * self.conn)), '{:,}'.format(total_size),
                    '{:,}'.format(self.n_pre), '{:,}'.format(self.n_post)
                    ), logfile)

            while size_left > max_size:
                start_time0 = time.time()

                # generate binary array
                conn_matrix_part = gen_binary_array(max_size, self.conn)
                non_zeros_part = count_step * max_size + np.nonzero(conn_matrix_part)[0]
                synapses[count_syn:count_syn + len(non_zeros_part)] = non_zeros_part
                count_syn += len(non_zeros_part)

                count_step += 1

                # estimate time left
                size_left = size_left - max_size
                curr_percent = (1 - size_left / total_size) * 100
                time_spent += (time.time() - start_time0)
                time_left = (100 - curr_percent) * time_spent / curr_percent
                pprint('\t\t %s: %.0f%% calculated in %.0f seconds (ca. %.0f seconds left...)' %
                       ('{:,}'.format(count_syn), curr_percent, time_spent, time_left), logfile)

        # generate binary array
        conn_matrix_part = gen_binary_array(size_left, self.conn)
        non_zeros_part = count_step * max_size + np.nonzero(conn_matrix_part)[0]
        synapses[count_syn:count_syn + len(non_zeros_part)] = non_zeros_part
        count_syn += len(non_zeros_part)

        # only keep generated synapses
        synapses = synapses[:count_syn]

        pprint('\t %s: calculated all synapses in %.0f seconds.' %
               ('{:,}'.format(count_syn), (time.time() - total_start_time)), logfile)

        # store synaptic indices
        self.pre_idx = np.array(synapses % self.n_pre, dtype=int)
        self.post_idx = np.array(synapses // self.n_pre, dtype=int)
        self.n_syn = len(self.pre_idx)


def gen_binary_array(array_size, prob):
    """
    generate boolean array where each element has probability 'prob' of being True
    'prob' precision must not be smaller than 1/integer_scale
    """

    integer_scale = 1000
    if int(prob*integer_scale) == 0:
        print('Error, probability precision is too small!')
        exit()

    rand_array = np.random.randint(integer_scale, size=array_size, dtype=np.uint16)
    return rand_array < integer_scale*prob


def s_to_hh_mm_ss(time_in):
    """
    convert seconds to a str with format hh:mm:ss
    """

    time_hour = time_in // 3600
    time_min = time_in % 3600 // 60
    time_sec = (time_in % 3600) % 60
    time_str = '%02d:%02d:%02d' % (time_hour, time_min, time_sec)
    return time_str


def get_logger(settings):
    """
    get name of log file
    """
    test_idx = settings['test_idx']
    file_name = settings['output_dir'] + settings['group_name'] + '_' +\
        settings['time_stamp'] + '/' + str(test_idx) + '.log'

    return file_name


def pprint(string, logfile=None):
    """
    print 'string' to the command line and to log file
    """
    print(string)

    if logfile is not None:
        file = open(logfile, 'a')
        file.write(string + '\n')
        file.close()


def gauss_window(x, sigma):
    """
    gaussian with zero mean and width sigma
    """
    return np.exp(-(x/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)


def smooth_array(array_, dt, smooth_window):
    """
    smooth an array by convolving it with a gaussian with a given window
    """
    if smooth_window / second > 0:
        t = np.arange(-3*smooth_window/dt, 3*smooth_window/dt + 1, 1)  # +/- 3*sigma
        array_ = np.convolve(array_, gauss_window(t, smooth_window / dt), mode='same')

    return array_


def calc_rate_monitor(spike_monitor, t_start, t_stop, dt, smooth_window=0):
    """
    calculate rate monitor from brian2 spiking monitor

    Args:
        spike_monitor: brian2 spiking monitor object
        t_start: calculation start time
        t_stop: calculation stop time
        dt: simulation step time
        smooth_window: time window to smooth population rate

    Returns:
        time_array: rate monitor time array
        rate_monitor: rate monitor array

    """
    n_cells = len(spike_monitor.spike_trains())

    spike_times = spike_monitor.t[(spike_monitor.t >= t_start) & (spike_monitor.t <= t_stop)]

    time_array = np.arange(t_start / second, (t_stop + dt) / second, dt/second)

    spike_counts = np.zeros_like(time_array)
    unique_times, unique_counts = np.unique(spike_times, return_counts=True)
    unique_idx = np.array((unique_times - t_start)/dt, dtype=int)
    spike_counts[unique_idx] = unique_counts

    rtm_array = spike_counts/((dt/second) * n_cells)

    # smooth rate monitor:
    rtm_array = smooth_array(rtm_array, dt, smooth_window)

    return time_array, rtm_array


def calc_unit_rates(spike_monitor, t_start, t_stop):
    """
    calculate individual firing rate for each neuron

    Args:
        spike_monitor: brian spike monitor object
        t_start: calculation start time
        t_stop: calculation stop time

    Returns:
        mean_rate: mean firing rate across all neurons
        std_rate: standard deviation of firing rate across all neurons

    """

    spike_trains = spike_monitor.spike_trains()
    num_neurons = len(spike_trains)
    num_spikes = np.zeros(num_neurons)

    for i in range(num_neurons):
        num_spikes[i] = len((spike_trains[i])[(spike_trains[i] >= t_start) &
                                              (spike_trains[i] <= t_stop)])

    rates = num_spikes / (t_stop - t_start)

    mean_rate = np.mean(rates)/Hz
    std_rate = np.std(rates)/Hz

    return mean_rate, std_rate


def lorentzian(x_array, a, mu, sigma):
    """
    the lorentzian function

    Args:
        x_array: argument of function
        a: parameter 1
        mu: parameter 2, peak center
        sigma: parameter 3, width at half maximum

    Returns:
        output: the lorentzian function
    """

    output = (a / pi) * sigma / ((x_array - mu) ** 2 + sigma ** 2)
    return output


def get_q_factor(a, sigma):
    """
    calculate Q-factor of a lorentzian

    Args:
        a: parameter 1 of Lorentzian function
        sigma: parameter 3 of Lorentzian function

    Returns:
        q_factor
    """

    peak = a / (pi * sigma)
    fwhm = 2 * sigma
    q_factor = peak / fwhm

    return q_factor


def calc_network_frequency(pop_rate, sim_time, dt, max_freq, fit=True):
    """
    calculate Power Spectral Density (PSD) of population activity
    and try to fit it to a lorentzian function

    Args:
        pop_rate: population rate signal
        sim_time: total time of pop_rate signal
        dt: time step of pop_rate signal
        max_freq: maximum frequency of the power spectrum
        fit: [true/false] if true, try to fit PSD to lorentzian

    Returns:
        fft_freq: frequency array of FFT
        fft_psd: PSD array of FFT
        net_freq: network frequency (None if lorentzian fit fails)
        q_factor: Q-factor (None if lorentzian fit fails)
    """

    # Power Spectral Density (PSD) (absolute value of Fast Fourier Transform) centered around the mean:
    fft_psd = np.abs(np.fft.fft(pop_rate - np.mean(pop_rate)) * dt) ** 2 / (sim_time / second)

    # frequency arrays for PSD:
    fft_freq = np.fft.fftfreq(pop_rate.size, dt)

    # delete second (mirrored) half of PSD:
    fft_psd = fft_psd[:int(fft_psd.size / 2)]
    fft_freq = fft_freq[:int(fft_freq.size / 2)]

    # find argument where frequency is closest to max_freq:
    arg_lim = (np.abs(fft_freq - max_freq)).argmin()

    # select power spectrum range [1,max_freq] Hz:
    fft_freq = fft_freq[1:arg_lim + 1]
    fft_psd = fft_psd[1:arg_lim + 1]

    fit_params = []
    if fit and (fft_psd != 0).any():

        # find maximum or power spectrum:
        i_arg_max = np.argmax(fft_psd)
        freq_max = fft_freq[i_arg_max]
        psd_max = fft_psd[i_arg_max]

        # fit power spectrum peak to Lorentzian function:
        try:
            fit_params, _ = scipy.optimize.curve_fit(lorentzian, fft_freq, fft_psd,
                                                     p0=(psd_max, freq_max, 1), maxfev=1000)
        except RuntimeError:
            pass
        finally:
            if fit_params is []:
                print("WARNING: Couldn't fit PSD to Lorentzian")

    # calculate network frequency and q-factor:
    net_freq = None
    q_factor = None
    if len(fit_params) > 0:
        lorentz_a, lorentz_mu, lorentz_sigma = fit_params
        if lorentz_mu >= 0:
            net_freq = lorentz_mu
            q_factor = get_q_factor(lorentz_a, lorentz_sigma)

    return fft_freq, fft_psd, net_freq, q_factor


def calc_isi_cv(spike_monitor, t_start, t_stop):
    """
    calculate the Coefficient of Variation (CV) of
    the Inter-Spike-Interval (ISI) for each neuron

    Args:
        spike_monitor: brian spike monitor
        t_start: start time of calculation
        t_stop: stop time of calculation

    Returns:
        mean_isi_cv: mean ISI CV across all neurons
        std_isi_cv: std of ISI CV across all neurons
        check_enough_spikes: [true/false] true if there are enough spikes for calculation

    """

    mean_isi_cv = 0
    std_isi_cv = 0

    # get spike times for whole network:
    spike_trains = spike_monitor.spike_trains()

    # trim spikes within calculation time range:
    num_neurons = len(spike_trains)
    cut_spike_trains = {new_list: [] for new_list in range(num_neurons)}
    for i in range(num_neurons):
        cut_spike_trains[i] = (spike_trains[i])[(spike_trains[i] >= t_start) &
                                                (spike_trains[i] <= t_stop)]

    # check if at least one neuron spikes at least twice in the selected interval:
    n = 0
    for i in range(num_neurons):
        if (cut_spike_trains[i]).size >= 2:
            n += 1

    # n is the number of neurons that spiked at least twice.
    # if there is at least one of those:
    if n > 0:
        check_enough_spikes = True

        # calculate ISI CV for each neuron:
        all_isi_cvs = np.zeros(n)
        j = 0
        for i in range(num_neurons):
            # if the neuron spiked at least twice:
            if (cut_spike_trains[i]).size >= 2:
                # get array of ISIs:
                isi = np.diff(cut_spike_trains[i])
                # calculate average and std of ISIs:
                avg_isi = np.mean(isi)
                std_isi = np.std(isi)
                # store value of neuron ISI CV
                all_isi_cvs[j] = std_isi / avg_isi
                j = j + 1

        # calculate mean and std of all ISI CVs:
        mean_isi_cv = np.mean(all_isi_cvs)
        std_isi_cv = np.std(all_isi_cvs)

    # if not enough spikes to perform calculation:
    else:
        check_enough_spikes = False

    return mean_isi_cv, std_isi_cv, check_enough_spikes


def check_brian_monitor(network, mon_name, mon_attr):
    """
    check if a given brian monitor with
    a given attribute (variable being measured) exists in the network object

    Args:
        network: brian network object
        mon_name: name of monitor to check
        mon_attr: name of monitor attribute to check

    Returns:
        check: [true/false]
    """

    check = False
    if mon_name in network:
        monitor = network[mon_name]
        if hasattr(monitor, mon_attr):
            check = True

    return check


def trim_brian_monitor(monitor, attr, attr_unit, t_start, t_stop):
    """
    trim a given brian monitor attribute to a given time range

    Args:
        monitor: brian monitor object
        attr: attribute of monitor (variable being measured)
        attr_unit: output unit of attribute
        t_start: start time of calculation
        t_stop: stop time of calculation

    Returns:
        time_array: trimmed time array
        attr_array: trimmed attribute array (unit-less)
    """

    time_array = np.array(monitor.t / second)[(monitor.t >= t_start) &
                                              (monitor.t <= t_stop)]

    attr_array = np.array(attr / attr_unit)[(monitor.t >= t_start) &
                                            (monitor.t <= t_stop)]

    return time_array, attr_array
