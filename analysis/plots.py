import matplotlib.pyplot as plt

from .metrics import average_accuracy


def generate_plots(results, plots_dir):
    plt.style.use('default')

    plot_task_1_for_all_experiments(results, show_std=False, ylim=(0, 100), plots_dir=plots_dir)
    plot_all_tasks_for_experiment(1, results, show_std=False, ylim=(0, 100), plots_dir=plots_dir)
    plot_all_tasks_for_experiment(4, results, show_std=False, ylim=(0, 100), plots_dir=plots_dir)
    plot_average_accuracy(results, ylim=(0, 100), plots_dir=plots_dir)


def plot_task_1_for_all_experiments(results, show_std, ylim, plots_dir):
    performances = [e.performances[0] for e in results]
    stds = None
    if show_std:
        try:
            stds = [e.stds[0] for e in results]
        except TypeError:
            stds = None
    exp_ns = [e.experiment_no for e in results]

    figure = plot_lines(
        performances,
        list_with_errors=stds,
        line_names=[f'Experiment {exp_n}' for exp_n in exp_ns],
        title=f"Performance on Task 1 throughout Experiment(s): {exp_ns}",
        ylabel="Test Accuracy (%) on Task 1",
        xlabel="Batch",
        figsize=(10,5),
        v_line=results[0].switch_indices[:-1],
        v_label='Task switch',
        ylim=ylim,
        save_as=f"{plots_dir}/experiments_{exp_ns}_task_1.svg"
    )


def plot_all_tasks_for_experiment(experiment_no, results, show_std, ylim, plots_dir):
    experiment = list(filter(lambda x: x.experiment_no == experiment_no, results))[0]
    performances = experiment.performances
    n_batches = experiment.switch_indices[-1]
    x_axes = [range(n_batches - len(p), n_batches) for p in performances]
    task_ns = list(range(1, len(performances) + 1))

    stds = None
    if show_std:
        try:
            stds = experiment.stds
        except TypeError:
            stds = None

    figure = plot_lines(
        performances,
        x_axes=x_axes,
        list_with_errors=stds,
        line_names=[f'Task {task_n}' for task_n in task_ns],
        title=f"Performance on each task throughout Experiment {experiment_no}",
        ylabel="Test Accuracy (%) on Task",
        xlabel="Batch",
        figsize=(10,5),
        v_line=experiment.switch_indices[:-1],
        v_label='Task switch', ylim=ylim,
        save_as=f"{plots_dir}/experiment_{experiment_no}_all_tasks.svg"
    )


def plot_average_accuracy(results, ylim, plots_dir):
    performances = [e.performances for e in results]
    exp_ns = [e.experiment_no for e in results]
    switch_indices = results[0].switch_indices

    avg_accuracies = [average_accuracy(perfs, switch_indices) for perfs in performances]

    figure = plot_lines(
        avg_accuracies,
        line_names=[f'Experiment {exp_n}' for exp_n in exp_ns],
        title=f"Average accuracy on all tasks throughout Experiment(s): {exp_ns}",
        ylabel="Average accuracy (%) on all tasks",
        xlabel="Batch",
        figsize=(10,5),
        v_line=switch_indices[:-1],
        v_label='Task switch', ylim=ylim,
        save_as=f"{plots_dir}/experiments_{exp_ns}_avg.svg"
    )


# Code taken, with some adjustments, from:
# https://github.com/GMvandeVen/continual-learning/blob/50b8b7fce9786dc402866fc8387e1525f369bbc5/visual/visual_plt.py#L103
def plot_lines(list_with_lines, x_axes=None, line_names=None, colors=None, title=None,
               title_top=None, xlabel=None, ylabel=None, ylim=None, figsize=None, list_with_errors=None, errors="shaded",
               x_log=False, with_dots=False, linestyle='solid', h_line=None, h_label=None, h_error=None,
               h_lines=None, h_colors=None, h_labels=None, h_errors=None,
               v_line=None, v_label=None, save_as=None, should_show=True):
    '''Generates a figure containing multiple lines in one plot.

    :param list_with_lines: <list> of all lines to plot (with each line being a <list> as well)
    :param x_axes:          <list> containing the values for the x-axis
    :param line_names:      <list> containing the names of each line
    :param colors:          <list> containing the colors of each line
    :param title:           <str> title of plot
    :param title_top:       <str> text to appear on top of the title
    :return: f:             <figure>
    '''

    # if needed, generate default x-axis
    if x_axes == None:
        n_obs = len(list_with_lines[0])
        x_axes = [list(range(n_obs)) for _ in range(len(list_with_lines))]

    # if needed, generate default line-names
    if line_names == None:
        n_lines = len(list_with_lines)
        line_names = ["line " + str(line_id) for line_id in range(n_lines)]

    # make plot
    size = (12,7) if figsize is None else figsize
    f, axarr = plt.subplots(1, 1, figsize=size)

    # add error-lines / shaded areas
    if list_with_errors is not None:
        for line_id, name in enumerate(line_names):
            if errors=="shaded":
                axarr.fill_between(
                    x_axes[line_id], 
                    list(np.array(list_with_lines[line_id]) + np.array(list_with_errors[line_id])),
                    list(np.array(list_with_lines[line_id]) - np.array(list_with_errors[line_id])),
                    color=None if (colors is None) else colors[line_id],
                    alpha=0.25)
            else:
                axarr.plot(
                    x_axes[line_id],
                    list(np.array(list_with_lines[line_id]) + np.array(list_with_errors[line_id])),
                    label=None,
                    color=None if (colors is None) else colors[line_id],
                    linewidth=1,
                    linestyle='dashed')
                axarr.plot(
                    x_axes[line_id],
                    list(np.array(list_with_lines[line_id]) - np.array(list_with_errors[line_id])),
                    label=None,
                    color=None if (colors is None) else colors[line_id],
                    linewidth=1,
                    linestyle='dashed')

    # mean lines
    for line_id, name in enumerate(line_names):
        axarr.plot(x_axes[line_id], list_with_lines[line_id], label=name,
                   color=None if (colors is None) else colors[line_id],
                   linewidth=4, marker='o' if with_dots else None, linestyle=linestyle if type(linestyle)==str else linestyle[line_id])

    # add horizontal line
    if h_line is not None:
        axarr.axhline(y=h_line, label=h_label, color="grey")
        if h_error is not None:
            if errors == "shaded":
                axarr.fill_between([x_axes[0], x_axes[-1]],
                                   [h_line + h_error, h_line + h_error], [h_line - h_error, h_line - h_error],
                                   color="grey", alpha=0.25)
            else:
                axarr.axhline(y=h_line + h_error, label=None, color="grey", linewidth=1, linestyle='dashed')
                axarr.axhline(y=h_line - h_error, label=None, color="grey", linewidth=1, linestyle='dashed')

    # add horizontal lines
    if h_lines is not None:
        h_colors = colors if h_colors is None else h_colors
        for line_id, new_h_line in enumerate(h_lines):
            axarr.axhline(y=new_h_line, label=None if h_labels is None else h_labels[line_id],
                          color=None if (h_colors is None) else h_colors[line_id])
            if h_errors is not None:
                if errors == "shaded":
                    axarr.fill_between([x_axes[0], x_axes[-1]],
                                       [new_h_line + h_errors[line_id], new_h_line+h_errors[line_id]],
                                       [new_h_line - h_errors[line_id], new_h_line - h_errors[line_id]],
                                       color=None if (h_colors is None) else h_colors[line_id], alpha=0.25)
                else:
                    axarr.axhline(y=new_h_line+h_errors[line_id], label=None,
                                  color=None if (h_colors is None) else h_colors[line_id], linewidth=1,
                                  linestyle='dashed')
                    axarr.axhline(y=new_h_line-h_errors[line_id], label=None,
                                  color=None if (h_colors is None) else h_colors[line_id], linewidth=1,
                                  linestyle='dashed')

    # add vertical line(s)
    if v_line is not None:
        if type(v_line)==list:
            for id,new_line in enumerate(v_line):
                axarr.axvline(x=new_line, label=v_label if id==0 else None, color='black')
        else:
            axarr.axvline(x=v_line, label=v_label, color='black')

    # finish layout
    # -set y-axis
    if ylim is not None:
        axarr.set_ylim(ylim)
    # -add axis-labels
    if xlabel is not None:
        axarr.set_xlabel(xlabel)
    if ylabel is not None:
        axarr.set_ylabel(ylabel)
    # -add title(s)
    if title is not None:
        axarr.set_title(title)
    if title_top is not None:
        f.suptitle(title_top)
    # -add legend
    if line_names is not None:
        axarr.legend()
    # -set x-axis to log-scale
    if x_log:
        axarr.set_xscale('log')
    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight')
    if should_show:
        plt.show()


    # return the figure
    return f
