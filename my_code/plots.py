from brian2 import *
from matplotlib import colors
from matplotlib import gridspec
from my_code.tests import Stimulus


def color_gradient(color_i, color_f, n):
    """
    calculates array of intermediate colors between two given colors

    Args:
        color_i: initial color
        color_f: final color
        n: number of colors

    Returns:
        color_array: array of n colors
    """

    if n > 1:
        rgb_color_i = np.array(colors.to_rgb(color_i))
        rgb_color_f = np.array(colors.to_rgb(color_f))
        color_array = [None] * n
        for i in range(n):
            color_array[i] = colors.to_hex(rgb_color_i*(1-i/(n-1)) + rgb_color_f*i/(n-1))
    else:
        color_array = [color_f]

    return color_array


def shade_stimulation(ax, t_start, t_dur, shade_color, shade_prop='1/1'):
    """
    shade portion of subplot to highlight where a stimulation happened

    Args:
        ax: subplot axis
        t_start: shade starting time
        t_dur: shade duration
        shade_color: shade color
        shade_prop: vertical proportion of subplot to shade
    """

    # calculate top and bottom of shading area:
    y_bottom, y_top = ax.get_ylim()

    y_height = y_top - y_bottom
    prop_numerator, prop_denominator = shade_prop.split('/')
    shade_bottom = y_bottom + y_height*(int(prop_numerator) - 1)/int(prop_denominator)
    shade_top = shade_bottom + y_height*1/int(prop_denominator)

    # create shading rectangle:
    vertices = [(t_start, shade_bottom),
                (t_start + t_dur, shade_bottom),
                (t_start + t_dur, shade_top),
                (t_start, shade_top)]
    rect = Polygon(vertices, facecolor=shade_color, edgecolor=shade_color, alpha=0.2)
    ax.add_patch(rect)


def get_height(max_height):
    """
    round maximum height and get y-axis ticks

    Args:
        max_height: maximum plot height

    Returns:
        max_height: rounded height
        y_ticks: y-axis ticks

    """
    if max_height > 10:
        y_ticks = [0, int(math.ceil(max_height / 20) * 10)]
    elif max_height > 5:
        y_ticks = [0, int(math.ceil(max_height / 10) * 5)]
    elif max_height > 1:
        y_ticks = [0, int(math.ceil(max_height / 4) * 2)]
    elif max_height > 0.5:
        y_ticks = [0.0, 0.5]
    elif max_height > 0.1:
        y_ticks = [0.0, 0.1]
    elif max_height > 0.05:
        y_ticks = [0, 0.05]
    elif max_height > 0.01:
        y_ticks = [0, 0.01]
    elif max_height > 0.005:
        y_ticks = [0, 0.005]
    else:
        y_ticks = [0, 0.001]

    if max_height == 0:
        y_ticks = [0]

    return max_height, y_ticks


class MySubplot:
    def __init__(self, subplot_name, monitor_names, plot_type, plot_colors, plot_params, height_ratio):
        """
        initialise subplot object

        Args:
            subplot_name: subplot name
            monitor_names: array with name of monitors to be plotted on this subplot
            plot_type: type of plot
                        'raster': raster plot
                        'trace': "normal" 2d plot
            plot_colors: array of colors for each monitor to be plotted on this subplot
            plot_params: plot parameters
            height_ratio: subplot height ratio
        """

        self.subplot_name = subplot_name
        self.monitor_names = monitor_names
        self.num_traces = len(self.monitor_names)
        self.plot_type = plot_type
        self.plot_colors = plot_colors
        self.ax = None
        self.lines = [None] * self.num_traces
        self.font_size = plot_params['text_font'].get_param()
        self.spine_width = plot_params['spine_width'].get_param()
        self.height_ratio = height_ratio

    def attr_ax(self, subplot_axes):
        self.ax = subplot_axes

    def set_title(self, title_text):
        self.ax.set_title(title_text, loc='left', x=0., y=1.02, fontsize=self.font_size)

    def hide_bottom(self):
        self.ax.spines['bottom'].set_visible(False)
        self.ax.set_xticks([])

    def hide_left(self):
        self.ax.spines['left'].set_visible(False)
        self.ax.set_yticks([])

    def show_time_labels(self):
        self.ax.set_xlabel('time (s)')

    def hide_time_labels(self):
        self.ax.set_xticklabels(labels=[])
        self.ax.set_xlabel('')

    def set_x_ticks(self, x_ticks):
        self.ax.set_xticks(x_ticks)
        self.ax.set_xticklabels(labels=x_ticks, fontsize=self.font_size)

    def set_y_ticks(self, y_ticks):
        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels(labels=y_ticks, fontsize=self.font_size)

    def set_y_label(self, label):
        self.ax.set_ylabel(label, fontsize=self.font_size)

    def set_time_ticks(self, plot_start, plot_stop, n_ticks):
        time_ticks = [round(plot_start + (plot_stop - plot_start) * i / n_ticks, 2) for i in
                      range(n_ticks + 1)]
        self.set_x_ticks(time_ticks)

    def add_lines(self):
        for i in range(self.num_traces):
            if self.plot_type == 'raster':
                self.lines[i], = self.ax.plot([], [], '.', ms=1.0, color=self.plot_colors[i])

            elif self.plot_type == 'trace':
                self.lines[i], = self.ax.plot([], [], lw=1, color=self.plot_colors[i])

    def general_format(self):
        self.ax.tick_params(top=False, which='both', labelsize=self.font_size, direction='in', width=self.spine_width)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # all raster plots include only 50*height_ratio neurons:
        if self.plot_type == 'raster':
            self.ax.set_ylim([-2, int(50*self.height_ratio) + 1])
            self.hide_left()

            # if more than one monitor in one raster plot (for assemblies), split the 50*height_ratio neurons equally:
            if self.num_traces > 1:
                asb_height = int(50 * self.height_ratio / self.num_traces)
                h_lines = np.arange(0, asb_height*(self.num_traces + 1), asb_height)
                for h in h_lines:
                    self.ax.axhline(h - 0.5, lw=0.5, color='lightgray', ls='solid')

        # pretty ticks:
        y_ticks = self.ax.get_yticks()
        self.set_y_ticks(y_ticks)
        x_ticks = self.ax.get_xticks()
        self.set_x_ticks(x_ticks)

    def set_lines(self, monitors):
        """
        attribute the correct monitor to each subplot line

        Args:
            monitors: dictionary containing all prepared brian monitors from test run
        """

        # iterate through each line to be plotted
        max_height = 0
        for i in range(self.num_traces):

            # if required monitor has been calculated:
            if self.monitor_names[i] in list(monitors.keys()):

                # multiple raster plots in one subplot are split equally:
                if (self.plot_type == 'raster') and (self.num_traces > 1):
                    num = int(50 * self.height_ratio / self.num_traces)
                    times, neurons = monitors[self.monitor_names[i]]
                    idx = neurons < num
                    shifted_neurons = neurons[idx] + i*num
                    monitor = times[idx], shifted_neurons
                else:
                    monitor = monitors[self.monitor_names[i]]

                # set monitor x and y arrays to subplot line
                self.lines[i].set_xdata(monitor[0])
                self.lines[i].set_ydata(monitor[1])

                # update y-axis for trace plots:
                if self.plot_type == 'trace':
                    if len(monitor[1]) > 0:
                        # get height of trace with the highest value:
                        height = np.max(monitor[1])
                        if height > max_height:
                            max_height = height

                        # round and get ticks
                        max_height, y_ticks = get_height(max_height)

                        # apply to axis
                        if max_height > 0:
                            self.ax.set_ylim([0, max_height * 1.05])
                        else:
                            self.ax.set_ylim([-1, 1])
                        self.set_y_ticks(y_ticks)


class MySubplotGroup:
    def __init__(self, group_name, group_title, y_label=None):
        """
        create group of subplots

        Args:
            group_name: subplot group name
            group_title: subplot group title
            y_label: subplot group y-axis label
        """

        self.group_name = group_name
        self.group_title = group_title
        self.time_labels = False
        self.y_label = y_label
        self.subplots = []

    def add_subplot(self, subplot_name, monitor_names, plot_type, plot_colors, plot_params, height_ratio=1.0):
        """
        add a subplot to this subplot group.
        each subplot within the group can have more than one monitor.

        Args:
            subplot_name: name of subplot
            monitor_names: array with name of monitors to be plotted
            plot_type: type of subplot
                        'raster': raster plot
                        'trace': "normal" 2d plot
            plot_colors: array with colors for each monitor to be plotted
            plot_params: plot parameters
            height_ratio: subplot height ratio
        """

        self.subplots.append(MySubplot(subplot_name, monitor_names, plot_type, plot_colors, plot_params, height_ratio))

    def init_group_format(self, time_labels=False):
        self.time_labels = time_labels
        self.subplots[0].set_title(self.group_title)

        # for every subplot in group:
        for i in range(len(self.subplots)):
            self.subplots[i].general_format()

            # hide bottom of all subplots in group except for the last:
            if i < len(self.subplots) - 1:
                self.subplots[i].hide_bottom()

        # if a y label was given, set it on the middle subplot of the group:
        if self.y_label is not None:
            label_idx = int(len(self.subplots) / 2)
            self.subplots[label_idx].set_y_label(self.y_label)

    def set_time_axes(self, plot_start, plot_stop, n_ticks):
        for i in range(len(self.subplots)):
            self.subplots[i].ax.set_xlim([plot_start, plot_stop])

        self.subplots[-1].set_time_ticks(plot_start, plot_stop, n_ticks)
        if self.time_labels:
            self.subplots[-1].show_time_labels()
        else:
            self.subplots[-1].hide_time_labels()


def create_test_figure(net_params, plot_params, test_data, monitors, events):
    """
    create figure where standard test results will be drawn.
    all subplots and brian monitors are specified here, but no data is attributed to them

    Args:
        net_params: network parameters
        plot_params: plot parameters
        test_data: calculated test data
        monitors: brian monitor data ready for plotting
        events: simulation events

    Returns:
        fig: matplotlib figure object
    """

    p_color = plot_params['p_color'].get_param()
    b_color = plot_params['b_color'].get_param()

    # initialise dictionary where all subplot groups will be stored:
    subplot_groups = {}

    """ Raster plots """
    group_idx = 0
    group_name = 'Spikes'
    subplot_groups[group_idx] = MySubplotGroup('spm', group_name)

    # P cells outside assemblies:
    subplot_groups[group_idx].add_subplot('spm_p_out', ['spm_p_out'], 'raster', ['dimgray'],
                                          plot_params, height_ratio=1.0)

    # P cells within assemblies:
    n_asb = int(net_params['n_asb'].get_param())
    colors_spm_p = color_gradient(p_color, 'darkred', n_asb)
    monitors_spm_p = []
    for i in range(n_asb):
        monitors_spm_p += ['spm_p_asb_' + str(i + 1)]
    subplot_groups[group_idx].add_subplot('spm_p', monitors_spm_p, 'raster', colors_spm_p,
                                          plot_params, height_ratio=2.0)

    # B cells:
    subplot_groups[group_idx].add_subplot('spm_b', ['spm_b'], 'raster', [b_color], plot_params)

    """ Population rate plots """
    group_idx += 1
    group_name = 'Population Rates'
    subplot_groups[group_idx] = MySubplotGroup('rtm', group_name, y_label=r'Rate (Hz)')

    # P cells outside assemblies:
    subplot_groups[group_idx].add_subplot('rtm_p_out', ['rtm_p_out'], 'trace', ['dimgray'], plot_params)

    # P cell assemblies:
    colors_rtm_p = color_gradient(p_color, 'darkred', n_asb)
    monitors_rtm_p = []
    for i in range(n_asb):
        monitors_rtm_p += ['rtm_p_asb_' + str(i + 1)]
    subplot_groups[group_idx].add_subplot('rtm_p', monitors_rtm_p, 'trace', colors_rtm_p, plot_params)

    # B cells:
    subplot_groups[group_idx].add_subplot('rtm_b', ['rtm_b'], 'trace', [b_color], plot_params)

    """ Determine figure contents """
    # create figure object:
    plot_width = plot_params['plot_width'].get_param() / (cmeter * 2.54)
    fig_height = plot_params['fig_height'].get_param() / (cmeter * 2.54)
    fig = plt.figure(figsize=(plot_width, fig_height))

    # get total number of subplot rows
    num_groups = len(subplot_groups)
    num_vert = 0
    for group_idx in subplot_groups:
        group = subplot_groups[group_idx]
        num_vert += len(group.subplots)

    # calculate array of height ratios for all subplots:
    height_ratios = []
    for group_idx in subplot_groups:
        group = subplot_groups[group_idx]
        for i in range(len(group.subplots)):
            height_ratios += [1.0 * group.subplots[i].height_ratio]

        # add gap between subplot groups
        if group_idx < num_groups - 1:
            height_ratios += [0.5]

    # create grid where all subplots will be added:
    num_plots = num_vert + num_groups - 1
    gs = gridspec.GridSpec(num_plots, 1, height_ratios=height_ratios)

    # add all subplots to grid:
    plot_idx = 0
    for group_idx in subplot_groups:
        group = subplot_groups[group_idx]
        for i in range(len(group.subplots)):
            group.subplots[i].attr_ax(fig.add_subplot(gs[plot_idx]))
            plot_idx += 1
        plot_idx += 1

    # initialise groups and attribute line objects to each subplot:
    for group_idx in subplot_groups:
        group = subplot_groups[group_idx]

        # initialise group
        if group_idx == (num_groups - 1):
            group.init_group_format(time_labels=True)
        else:
            group.init_group_format(time_labels=False)

        # attribute line objects to subplots
        for i in range(len(group.subplots)):
            group.subplots[i].add_lines()

    plt.subplots_adjust(hspace=0.15)

    """ Draw figure contents """

    plot_start = test_data['calc_start']
    plot_stop = test_data['calc_stop']
    n_time_ticks = plot_params['num_time_ticks'].get_param()

    for group_idx in subplot_groups:
        group = subplot_groups[group_idx]
        group.set_time_axes(plot_start / second, plot_stop / second, n_time_ticks)

        for sub_plot in group.subplots:

            # attribute brian monitors to plot lines:
            sub_plot.set_lines(monitors)

            # shade stimulations:
            for key_ in events:
                if isinstance(events[key_], Stimulus):
                    stimulus = events[key_]
                    if (sub_plot.subplot_name == 'spm_' + stimulus.target) and \
                            ('out' not in sub_plot.subplot_name) and stimulus.shade:

                        n_asb = int(sub_plot.num_traces - 1)
                        if (n_asb > 0) and stimulus.asb > 0:
                            frac_stim = str(stimulus.asb) + '/' + str(n_asb + 1)
                        else:
                            frac_stim = '1/1'

                        shade_color = 'gold'
                        shade_stimulation(sub_plot.ax, stimulus.onset / second, stimulus.length / second,
                                          shade_color, frac_stim)
    return fig
