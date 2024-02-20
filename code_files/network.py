from brian2 import *
import numpy as np
import scipy.stats
from code_files.aux_functions import ConnectivityMatrix, pprint
from code_files.parameters import Parameter


def build_network(net_params, logfile=None):
    """
    build brian network
    P cells represent excitatory pyramidal cells
    B cells represent inhibitory interneurons

    Args:
        net_params: network parameters
        logfile: log file

    Returns:
        built_network: brian network object
        net_params: network parameters with used flags

    """

    """ NEURON MODEL """

    neuron_eqs = '''
        dv/dt = (curr_leak + curr_syn + curr_adapt + curr_bg + curr_stim)/mem_cap : volt (unless refractory)
        curr_leak = g_leak*(e_rest - v) : amp
        curr_syn = curr_p + curr_b : amp
        mem_cap : farad
        g_leak : siemens
        e_rest : volt
        v_thres : volt
        v_reset : volt
        tau_refr : second
        curr_bg : amp
        curr_stim : amp
    '''
    curr_p_eqs = '''
        curr_p = g_p*(e_p - v) : amp
        e_p : volt
        dg_p/dt = -g_p/tau_d_p : siemens
        tau_d_p : second
    '''
    curr_b_eqs = '''
        curr_b = g_b * (e_b - v): amp
        e_b : volt
        dg_b / dt = -g_b / tau_d_b: siemens
        tau_d_b: second
    '''

    adapt_eqs = '''
        dw / dt = - w / tau_w : amp
        curr_adapt = - w : amp
        tau_w : second
        w_jump : amp
    '''

    adapt_update = '''
        v = v_reset
        w += w_jump
    '''

    non_adapt_eqs = '''
        curr_adapt = 0 * amp : amp
    '''

    """ CREATE CELL POPULATIONS """

    all_neurons = []

    # create P (excitatory) cell population:
    p_neuron_eqs = neuron_eqs + curr_p_eqs + curr_b_eqs
    p_neuron_eqs += adapt_eqs
    p_reset = adapt_update
    n_p = int(net_params['n_p'].get_param())
    pop_p = NeuronGroup(n_p, model=p_neuron_eqs,
                        threshold='v > v_thres', reset=p_reset,
                        refractory='tau_refr', method='euler',
                        name='pop_p')
    all_neurons.append(pop_p)

    # create B (inhibitory) cell population:
    b_neuron_eqs = neuron_eqs + curr_p_eqs + curr_b_eqs
    b_neuron_eqs += non_adapt_eqs
    b_reset = 'v = v_reset'
    n_b = int(net_params['n_b'].get_param())
    pop_b = NeuronGroup(n_b, model=b_neuron_eqs,
                        threshold='v > v_thres', reset=b_reset,
                        refractory='tau_refr', method='euler',
                        name='pop_b')
    all_neurons.append(pop_b)

    # assign parameters to each cell type:
    for pop in all_neurons:
        if pop.name == 'pop_p':
            pop.curr_bg = net_params['curr_bg_p'].get_param()
            pop.tau_w = net_params['p_tau_w'].get_param()
            pop.w_jump = net_params['p_w_jump'].get_param()
        elif pop.name == 'pop_b':
            pop.curr_bg = net_params['curr_bg_b'].get_param()

        pop.mem_cap = net_params['mem_cap'].get_param()
        pop.g_leak = net_params['g_leak'].get_param()
        pop.e_rest = net_params['e_rest'].get_param()
        pop.v_thres = net_params['v_thres'].get_param()
        pop.v_reset = net_params['v_reset'].get_param()
        pop.tau_refr = net_params['tau_refr'].get_param()
        pop.e_p = net_params['e_p'].get_param()
        pop.e_b = net_params['e_b'].get_param()

    # store all Brian objects created so far:
    built_network = Network(collect())

    pprint('Connecting network', logfile)

    conn_dict = {}
    connect_populations(net_params, built_network, conn_dict, 'pp', logfile=logfile)
    embed_sequence(net_params, built_network, conn_dict, 'pp', logfile)

    connect_populations(net_params, built_network, conn_dict, 'bp', logfile=logfile)
    connect_populations(net_params, built_network, conn_dict, 'pb', plastic=True, logfile=logfile)
    connect_populations(net_params, built_network, conn_dict, 'bb', logfile=logfile)

    pprint('================ Network Built ================', logfile)

    return built_network


def connect_populations(net_params, built_network, conn_dict, str_ij, plastic=False, logfile=None):
    """
    connects two neuron populations

    Args:
        net_params: network parameters
        built_network: brian network object
        conn_dict: dictionary where connectivity matrices are stored
        str_ij: name of connection (J -> I)
        plastic: [True/False], whether synapses are plastic, as in Vogels et al. (2011)
        logfile: log file

    """

    # connect population J to population I:
    str_i = str_ij[0]
    str_j = str_ij[1]
    pop_i = built_network['pop_' + str_i]
    pop_j = built_network['pop_' + str_j]
    n_i = pop_i.N
    n_j = pop_j.N

    # set decay time constant of synaptic conductance
    setattr(pop_i, 'tau_d_' + str_j, net_params['tau_d_'+str_ij].get_param())

    # initialise pseudo-random seed
    conn_seed = net_params['conn_seed'].get_param()
    np.random.seed(conn_seed)

    pprint('%s->%s:' % (str_j.upper(), str_i.upper()), logfile)
    p_ij = net_params['p_' + str_ij].get_param()
    if p_ij > 0:
        # calculate connectivity matrix
        conn_ij = ConnectivityMatrix(p_ij, n_j, n_i)
        conn_ij.create_conn(conn_seed, logfile=logfile)
        net_params['n_syn_' + str_ij] = Parameter(len(conn_ij.pre_idx))
        conn_dict['conn_' + str_ij] = conn_ij

        # get synaptic parameters as strings
        g_ij_str = net_params['g_' + str_ij].get_str()
        tau_l_ij = net_params['tau_l_' + str_ij].get_param()

        # non-plastic synaptic model equations
        if not plastic:
            model_eqs = 'g_%s : siemens' % str_ij
            on_pre_eqs = 'g_%s += g_%s' % (str_j, str_ij)
            on_post_eqs = ''

        # plastic synapses, as in Vogels et al. (2011)
        else:

            # get STDP parameters as strings
            rho0_str = net_params[str_ij + '_stdp_rho0'].get_str()  # desired firing rate of postsynaptic neurons
            tau_stdp_str = net_params[str_ij + '_stdp_tau'].get_str()  # time constant
            g_max_str = net_params[str_ij + '_stdp_g_max'].get_str()  # maximum weight
            eta_str = net_params[str_ij + '_stdp_eta'].get_str()  # learning rate

            # synaptic model equations
            model_eqs = '''
                           g_%s : siemens
                           eta_switch : 1
                           eta = %s  * eta_switch: 1
                           alpha = 2 * (%s) * (%s) : 1
                           dx_%s / dt = -x_%s / (%s) : 1 (event-driven)
                           dx_%s / dt = -x_%s / (%s) : 1 (event-driven)
                        ''' % (str_ij,
                               eta_str,
                               rho0_str, tau_stdp_str,
                               str_j, str_j, tau_stdp_str,
                               str_i, str_i, tau_stdp_str)

            # model what happens after a pre-synaptic spike
            on_pre_eqs = '''
                            x_%s += 1.
                            g_%s = clip(g_%s + (x_%s - alpha)*%s*eta, 0*nS, %s)
                            g_%s_post += g_%s
                         ''' % (str_j,
                                str_ij, str_ij, str_i, g_ij_str, g_max_str,
                                str_j, str_ij)

            # model what happens after a post-synaptic spike
            on_post_eqs = '''
                             x_%s += 1.
                             g_%s = clip(g_%s + x_%s*%s*eta, 0*nS, %s)
                          ''' % (str_i,
                                 str_ij, str_ij, str_j, g_ij_str, g_max_str)

        # build synapse object
        syn_ij = Synapses(pop_j, pop_i,
                          model=model_eqs,
                          on_pre=on_pre_eqs,
                          on_post=on_post_eqs,
                          delay=tau_l_ij,
                          method='euler',
                          name='syn_' + str_ij)
        syn_ij.connect(i=conn_ij.pre_idx, j=conn_ij.post_idx)

        # initialise conductance:
        setattr(syn_ij, 'g_%s' % str_ij, g_ij_str)

        # for plastic synapses, initialise learning rate at 0
        if plastic == 'stdp':
            setattr(syn_ij, 'eta_switch', 0)

        built_network.add(syn_ij)

    else:
        pprint('\t no connections created; probability is 0', logfile)
        return 0


def embed_sequence(net_params, built_network, conn_dict, str_ij, logfile=None):
    """
    1) creates a log-normal distribution of synaptic weights
    2) embeds a sequence of assemblies by strengthening the connections within and across subgroups of neurons

    Args:
        net_params: network parameters
        built_network: brian network object
        conn_dict: dictionary with connectivity matrices
        str_ij: name of connection (J -> I)
        logfile: log file

    """

    # get name of pre- and post-synaptic populations
    str_j = str_ij[1]
    str_i = str_ij[0]

    # get connectivity matrix:
    conn_ij = conn_dict['conn_' + str_ij]

    # create log-normal weight distribution
    weight_median = net_params['g_' + str_ij + '_median'].get_param()
    weight_pctl = net_params['g_' + str_ij + '_pctl'].get_param()
    weight_pctl_val = net_params['g_' + str_ij].get_param()
    mu_x = np.log(weight_median / nsiemens)
    sigma_x = np.log(weight_pctl_val / weight_median) / scipy.stats.norm.ppf(weight_pctl)
    weights = np.random.lognormal(mu_x, sigma_x, size=conn_ij.n_syn) * nsiemens

    # get number of assemblies and number of cells per assembly
    n_asb = net_params['n_asb'].get_param()
    n_i_asb = int(net_params['n_' + str_i + '_asb'].get_param())
    n_j_asb = int(net_params['n_' + str_j + '_asb'].get_param())

    # create assemblies:
    count_imposed = 0
    for i in range(n_asb):
        # recurrent connections:
        pprint('%s->%s (assembly %d):' % (str_j.upper(), str_i.upper(), i + 1), logfile)

        # select synapses that fall within rc range:
        rc_idx = np.argwhere((conn_ij.pre_idx >= i * n_j_asb) & (conn_ij.pre_idx < (i + 1) * n_j_asb) &
                             (conn_ij.post_idx >= i * n_i_asb) & (conn_ij.post_idx < (i + 1) * n_i_asb))

        # strengthen those synapses:
        weights[rc_idx] = net_params['g_%s' % str_ij].get_param()
        count_imposed += len(rc_idx)
        pprint('\t selected %s synapses' % f'{len(rc_idx):,}', logfile)

        # feedforward connections:
        ff_pre_idx = i
        ff_post_idx = i + 1
        if i == n_asb - 1:
            break

        pprint('%s (asb %d) -> %s (asb %d):' % (str_j.upper(), ff_pre_idx + 1,
                                                str_i.upper(), ff_post_idx + 1), logfile)

        # select synapses that fall within ff range:
        ff_idx = np.argwhere((conn_ij.pre_idx >= ff_pre_idx * n_j_asb) &
                             (conn_ij.pre_idx < (ff_pre_idx + 1) * n_j_asb) &
                             (conn_ij.post_idx >= ff_post_idx * n_i_asb) &
                             (conn_ij.post_idx < (ff_post_idx + 1) * n_i_asb))

        # strengthen those synapses:
        weights[ff_idx] = net_params['g_%s' % str_ij].get_param()
        count_imposed += len(ff_idx)
        pprint('\t selected %s synapses' % f'{len(ff_idx):,}', logfile)

    # attribute weights to synapses
    g_ij = getattr(built_network['syn_' + str_ij], 'g_%s' % str_ij)
    g_ij[:] = weights

    pprint('%s->%s: imposed %s / %s (%.4f %%) synapses' %
           (str_j.upper(), str_i.upper(),
            f'{count_imposed:,}', f'{len(weights):,}', 100 * count_imposed / len(weights)),
           logfile)


def init_network(built_network, net_params, test_params):
    """
    initialise network in a pseudo-random state

    Args:
        built_network: built brian network
        net_params: parameters used to build network
        test_params: test parameters

    Returns:
        built_network: initialised network
        test_params: test parameters (for tracking used params)

    """

    test_seed = int(test_params['test_seed'].get_param())
    np.random.seed(test_seed)

    v_reset = net_params['v_reset'].get_param()
    v_thres = net_params['v_thres'].get_param()

    pop_p = built_network['pop_p']
    pop_b = built_network['pop_b']

    pop_p.v = v_reset + (v_thres - v_reset) * np.random.rand(pop_p.N)
    pop_b.v = v_reset + (v_thres - v_reset) * np.random.rand(pop_b.N)

    # initialise synaptic conductances:
    pop_p.g_p = 0.1 * nS * np.random.rand(pop_p.N)
    pop_p.g_b = 0.1 * nS * np.random.rand(pop_p.N)
    pop_b.g_p = 0.1 * nS * np.random.rand(pop_b.N)
    pop_b.g_b = 0.1 * nS * np.random.rand(pop_b.N)


def record_network(built_network, net_params, test_params):
    """
    create monitors to record from network

    Args:
        built_network: built brian network
        net_params: parameters used to build network
        test_params: test parameters

    Returns:
        built_network: network with added monitors
        test_params: test parameters (for tracking used params)

    """
    test_seed = int(test_params['test_seed'].get_param())
    max_record = int(test_params['max_record'].get_param())

    """ MONITOR P CELLS """

    pop_p = built_network['pop_p']

    if pop_p.N > max_record:
        spm_p = SpikeMonitor(pop_p, record=False, name='spm_p')
    else:
        spm_p = SpikeMonitor(pop_p, name='spm_p')
    built_network.add(spm_p)

    # assembly monitors:
    n_asb = int(net_params['n_asb'].get_param())
    n_p_asb = int(net_params['n_p_asb'].get_param())
    spm_p_asb = []
    for i in range(n_asb):
        spm_p_asb.append(SpikeMonitor(pop_p[i*n_p_asb:(i+1)*n_p_asb], name='spm_p_asb_'+str(i+1)))
        built_network.add(spm_p_asb[i])

    # neurons outside assemblies:
    end_rec = n_asb*n_p_asb + max_record
    if end_rec > pop_p.N:
        end_rec = -1
    spm_p_out = SpikeMonitor(pop_p[n_asb*n_p_asb:end_rec], name='spm_p_out')
    built_network.add(spm_p_out)

    """ MONITOR B CELLS """
    pop_b = built_network['pop_b']

    if pop_b.N > max_record:
        spm_b = SpikeMonitor(pop_b[:max_record], name='spm_b')
    else:
        spm_b = SpikeMonitor(pop_b, name='spm_b')
    built_network.add(spm_b)

    # record g_pb from n random B->P synapses:
    syn_pb = built_network['syn_pb']
    n_syn_pb = int(net_params['n_syn_pb'].get_param())
    if n_syn_pb > max_record:
        rec_pb_idx2 = np.random.default_rng(test_seed).choice(n_syn_pb, size=max_record,
                                                              replace=False)
        stm_stdp_pb_g = StateMonitor(syn_pb, 'g_pb', record=rec_pb_idx2, name='stm_stdp_pb_g')
    else:
        stm_stdp_pb_g = StateMonitor(syn_pb, 'g_pb', record=True, name='stm_stdp_pb_g')
    built_network.add(stm_stdp_pb_g)

    stm_stdp_pb_eta = StateMonitor(syn_pb, 'eta', record=1, name='stm_stdp_pb_eta')
    built_network.add(stm_stdp_pb_eta)
