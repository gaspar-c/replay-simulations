from brian2 import *
from code_files.aux_functions import pprint


class Parameter:

    def __init__(self, val, pref_unit=1):
        """
        initialise parameter object, attributing it a quantity and preferred unit
        the parameter is marked as not used

        Args:
            val: parameter value
            pref_unit: preferred parameter unit
        """

        if type(val) is Quantity:
            print('ERROR: Parameter value should not have units')
        if type(val) is bool:
            self.quantity = val
        else:
            self.quantity = val * pref_unit
        self.pref_unit = pref_unit
        self.used = False

    def get_param(self):
        """
        output parameter quantity and mark parameter as used

        Returns:
            self.quantity: parameter quantity
        """

        self.used = True
        return self.quantity

    def get_str(self, use=True):
        """
        output parameter quantity as string in preferred unit and mark parameter as used

        Returns:
            output_string: string of parameter quantity in preferred unit
        """

        if use:
            self.used = True

        pref_unit_str = str(self.pref_unit)
        if '1. ' in pref_unit_str:
            pref_unit_str = pref_unit_str[3:]

        output_string = str(self.quantity / self.pref_unit) + ' * ' + pref_unit_str
        return output_string


def get_dft_net_params():
    """
    dictionary with default network parameters
    """

    # pseudo random seed for network connectivity:
    dft_net_params = {'conn_seed': Parameter(1)}

    # LIF neuron:
    dft_net_params = {**dft_net_params,
                      **{'mem_cap': Parameter(200, pF), 'g_leak': Parameter(10, nS),
                         'e_rest': Parameter(-60, mV), 'v_thres': Parameter(-50, mV),
                         'v_reset': Parameter(-60, mV), 'tau_refr': Parameter(1, ms)}
                      }

    # neuron populations:
    dft_net_params = {**dft_net_params,
                      **{'n_p': Parameter(20000), 'e_p': Parameter(0, mV), 'curr_bg_p': Parameter(200, pA),
                         'n_b': Parameter(5000), 'e_b': Parameter(-80, mV), 'curr_bg_b': Parameter(200, pA)}
                      }

    # connectivities:
    dft_net_params = {**dft_net_params,
                      **{'p_pp': Parameter(0.08), 'p_bp': Parameter(0.01),
                         'p_pb': Parameter(0.04), 'p_bb': Parameter(0.04)}
                      }

    # conductances:
    dft_net_params = {**dft_net_params,
                      **{'g_pp': Parameter(0.05, nS), 'g_bp': Parameter(0.05, nS),
                         'g_pb': Parameter(0.20, nS), 'g_bb': Parameter(0.20, nS),
                         'gamma': Parameter(1)}
                      }

    # synaptic decay:
    dft_net_params = {**dft_net_params,
                      **{'tau_d_pp': Parameter(2.0, ms), 'tau_d_bp': Parameter(2.0, ms),
                         'tau_d_pb': Parameter(4.0, ms), 'tau_d_bb': Parameter(4.0, ms)}
                      }

    # synaptic latency:
    dft_net_params = {**dft_net_params,
                      **{'tau_l_pp': Parameter(1, ms), 'tau_l_bp': Parameter(1, ms),
                         'tau_l_pb': Parameter(1, ms), 'tau_l_bb': Parameter(1, ms)}
                      }

    # adaptation parameters:
    dft_net_params = {**dft_net_params,
                      **{'p_tau_w': Parameter(20, ms), 'p_w_jump': Parameter(100, pA)}
                      }

    # P sequence parameters:
    dft_net_params = {**dft_net_params,
                      **{'n_asb': Parameter(10),
                         'n_p_asb': Parameter(1000),
                         'g_pp_median': Parameter(0.001, nS),
                         'g_pp_pctl': Parameter(0.99),
                         }
                      }

    # B->P STDP, as in Vogels et al. (2011):
    dft_net_params = {**dft_net_params,
                      **{'pb_stdp_rho0': Parameter(5, Hz),
                         'pb_stdp_tau': Parameter(20, ms), 'pb_stdp_g_max': Parameter(100, nS),
                         'pb_stdp_eta': Parameter(0.01)}
                      }

    return dft_net_params


def get_dft_test_params():
    """
    dictionary with default test parameters
    """

    # simulation parameters:
    dft_test_params = {'sim_dt': Parameter(0.10, ms),
                       'test_seed': Parameter(100),
                       'sim_time': Parameter(0, second)}

    # test calculations:
    dft_test_params = {**dft_test_params,
                       **{'max_record': Parameter(1000),
                          'smooth_window': Parameter(1, ms),
                          'isi_thres_irr': Parameter(0.5),
                          'isi_thres_reg': Parameter(0.3),
                          'q_factor_thres': Parameter(0.1),
                          'max_net_freq': Parameter(250, Hz)}
                       }

    # plasticity learning time:
    dft_test_params = {**dft_test_params,
                       **{'pb_stdp_start': Parameter(0, second),
                          'pb_stdp_dur': Parameter(10, second)}}

    return dft_test_params


def get_dft_plot_params():
    """
    dictionary with default plot parameters
    """

    dft_plot_params = {'plot_width': Parameter(20, cmeter),
                       'fig_height': Parameter(20, cmeter),
                       'p_color': Parameter('#ef3b53'),
                       'b_color': Parameter('dodgerblue'),
                       'text_font': Parameter(9),
                       'spine_width': Parameter(1.0),
                       'num_time_ticks': Parameter(5),
                       }

    return dft_plot_params


def load_specified_params(base_params, new_params, logfile=None):
    """
    store "new_params" in the "base_params" dictionary

    Args:
        base_params: dictionary with base parameters
        new_params: dictionary with new parameters
        logfile: log file name

    """
    for key_ in new_params.keys():
        param_k = new_params[key_]
        if key_ in base_params:
            if type(param_k) is tuple:
                base_params[key_] = Parameter(*param_k)
            else:
                base_params[key_] = Parameter(param_k)
            pprint('\t %s = %s' % (key_, new_params[key_]), logfile)
        else:
            pprint('\t Error: param %s does not exist' % key_, logfile)
