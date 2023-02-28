from brian2 import *
import numpy as np
from matplotlib import pyplot as plt, rc
import scipy.optimize
import pandas as pd

rc('text', usetex=True)


def group_plot(group_path,
               name_x, name_y, xlabel_, ylabel_,
               title_='', scale_x=1, scale_y=1, skip_xlabel=2, skip_ylabel=2,
               xlabel_append='', fit_inv=False):
    """
    plot results of a group test (for panels C, D, E)

    Args:
        group_path: folder path of test group
        name_x: name of variable to plot on the x-axis
        name_y: name of variable to plot on the y-axis
        xlabel_: x-axis label
        ylabel_: y-axis label
        title_: plot title
        scale_x: scale of x-axis variable
        scale_y: scale of y-axis variable
        skip_xlabel: skip labels in x-axis
        skip_ylabel: skip labels in y-axis
        xlabel_append: append str to x label
        fit_inv: fit x.y=cte line to the test results

    """
    # read group results file
    n_sample, final = read_data(group_path, name_x, name_y)

    # scale values
    final[name_x] = final[name_x] * scale_x
    final[name_y] = final[name_y] * scale_y

    # sort values
    val_x = np.sort(np.unique(final[name_x].values))
    val_y = np.sort(np.unique(final[name_y].values))

    # get percentage of successful replays for each (x,y) value pair
    xx, yy = np.meshgrid(val_x, val_y, indexing='xy')
    zz = np.zeros_like(xx)
    for i in range(len(val_x)):
        for j in range(len(val_y)):
            z = final.loc[(final[name_x] == val_x[i]) & (final[name_y] == val_y[j])]['quality'].values * 100
            if len(z) == 1:
                zz[j, i] = z[0]

    dx = val_x[1] - val_x[0]
    x_coord = np.append(val_x, val_x[-1] + dx) - dx / 2

    dy = val_y[1] - val_y[0]
    y_coord = np.append(val_y, val_y[-1] + dy) - dy / 2

    # plot results
    fig_height = 10 / 2.54
    fig_width = fig_height * 1.20
    font_size = 20
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # color bar with percentage of success
    color_map = 'YlGnBu'
    color_plot = ax.pcolormesh(x_coord, y_coord, zz, cmap=color_map)
    cbar = fig.colorbar(color_plot, ax=ax)
    bar_ticks = cbar.ax.get_yticks()
    cbar.set_ticks(bar_ticks)
    cbar.set_ticklabels(bar_ticks.astype(int))
    cbar.ax.tick_params(length=3, width=1, labelsize=font_size)
    cbar.set_label(r'\% ' + (r'Successful Replays (n=%d)' % n_sample),
                   size=font_size, rotation=-90, va='top', labelpad=20)

    # store axis limits
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    # fit line to threshold between unsuccessful and successful events
    if fit_inv:
        # find threshold
        threshold = np.zeros((0, 2))
        for y in val_y:
            for x in val_x:
                qual = final.loc[(final[name_x] == x) & (final[name_y] == y)]['quality'].values
                if qual > 0.20:
                    threshold = np.append(threshold, [[x, y]], axis=0)
                    break

        xdata = threshold[:, 0]
        ydata = threshold[:, 1]

        # fit an x.y=cte line to test results
        function = inverse_fit
        init_est = 50
        fit_params, pcov = scipy.optimize.curve_fit(function, xdata, ydata, p0=init_est, maxfev=10000)
        residuals = ydata - function(xdata, *fit_params)

        # find r2 of fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print('r2 = %f' % r_squared)
        print('params = %s' % fit_params)
        dx = val_x[1] - val_x[0]
        xrange = np.arange(val_x[0], val_x[-1] + dx, dx / 10)

        # plot fitted line
        ax.plot(xrange, function(xrange, *fit_params), c='white', ls='--', lw=2)

    # apply stored limits
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    # edit axes ticks
    ax.tick_params(length=3, width=1, direction='in', labelsize=font_size)
    ax.set_xticks(val_x)
    x_labels = []
    for i in range(len(val_x)):
        if i % skip_xlabel == 0:
            x_labels += [f'{int(val_x[i]):,}' + xlabel_append]
        else:
            x_labels += ['']
    ax.set_xticklabels(labels=x_labels, fontsize=font_size)

    ax.set_yticks(val_y)
    y_labels = []
    for i in range(len(val_y)):
        if i % skip_ylabel == 0:
            y_labels += [f'{int(val_y[i]):,}']
        else:
            y_labels += ['']
    ax.set_yticklabels(labels=y_labels, fontsize=font_size)

    # apply stored limits
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    # set aspect ratio
    aspect = (x_lims[1] - x_lims[0])/(y_lims[1] - y_lims[0])
    ax.set_aspect(aspect*len(val_y)/len(val_x))

    # set labels
    ax.set_xlabel(xlabel_, fontsize=font_size)
    ax.set_ylabel(ylabel_, fontsize=font_size)
    ax.set_title(title_, fontsize=font_size, loc='right', va='bottom', pad=1)

    # save results figure
    plt.savefig(group_path + '_results.png', dpi=600, bbox_inches='tight')


def read_data(group_path, var1, var2):
    """
    read data from group results file and calculate quality of replay (% of success)

    Args:
        group_path: folder path of test group
        var1: name of first variable
        var2: name of second variable

    Returns:
        max_trials: number of trials for a given variable pair
        replay_pivot: pivot table with replay quality

    """

    # read test results file and create a DataFrame
    file_path = group_path + '/0_group_results.txt'
    file = open(file_path, 'r')
    data = []
    for line in file.readlines():
        data.append(line.replace('\n', '').split(' \t '))
    file.close()
    df = pd.DataFrame(data=data[1:], columns=data[0])

    # convert from string to values
    df[var1] = val_from_str(df[var1])
    df[var2] = val_from_str(df[var2])

    # create pivot table that counts successful replays for each (var1, var2) pair
    df['replay'] = bool_from_str(df['replay']).astype(int)
    replay_pivot = df.pivot_table(values='replay', columns=[var1, var2], aggfunc={np.sum, 'size'}).T

    # check that all tests are complete
    counts_array = replay_pivot['size'].values
    max_trials = np.max(counts_array)
    equal_ = np.all(counts_array == max_trials)
    if not equal_:
        print('ERROR Not all tests are complete!')
        replay_array = replay_pivot['sum'].values
        if np.all(replay_array[counts_array != max_trials] == 0):
            print('All incomplete tests have quality 0! printing...')
        else:
            exit()

    # calculate replay quality (percentage of success)
    replay_pivot['quality'] = replay_pivot['sum'] / max_trials
    replay_pivot.drop(labels='sum', axis=1, inplace=True)
    replay_pivot.reset_index(inplace=True)
    print(replay_pivot)

    return max_trials, replay_pivot


def bool_from_str(str_array):
    """
    convert string array to boolean array
    """

    bool_array = np.zeros_like(str_array, dtype=bool)
    for k in range(len(str_array)):
        if str_array[k] == 'True':
            bool_array[k] = True
        elif str_array[k] == 'False':
            bool_array[k] = False
        else:
            print('ERROR str not True/False!')
            exit()

    return bool_array


def val_from_str(str_array, unit=1):
    """
    convert string array to value array
    """
    np_str = np.array(str_array, dtype=str)
    star_position = np.char.find(np_str, '*')
    val_array = np.zeros_like(np_str, dtype=float) * unit

    for k in range(len(str_array)):
        val_array[k] = float(np_str[k][:star_position[k]])

    return val_array


def inverse_fit(x_array, k):
    """
    y = k/x line
    """
    output = k/x_array
    return output
