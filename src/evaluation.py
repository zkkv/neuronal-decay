

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import json
    from training import ExperimentResult


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Evaluation""")
    return


@app.cell
def _():
    out_dir = "./out"
    results_file = f"{out_dir}/results/results.json"
    return out_dir, results_file


@app.cell
def _():
    displayed = [1, 2, 3, 4]

    print(f"[INFO] Displaying results only for experiments: {displayed}")
    return (displayed,)


@app.cell
def _(displayed, load_results_from_file, results_file):
    results = load_results_from_file(results_file)
    results = list(filter(lambda x: x.experiment_no in displayed, results))
    return (results,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Visualization""")
    return


@app.cell
def _(plot_all_tasks_for_experiment, plot_task_1_for_all_experiments, results):
    plt.style.use('default')

    plot_task_1_for_all_experiments(results, ylim=(0, 100))
    plot_all_tasks_for_experiment(1, results, ylim=(0, 100))
    plot_all_tasks_for_experiment(4, results, ylim=(0, 100))
    return


@app.cell(hide_code=True)
def _(plot_lines):
    def plot_task_1_for_all_experiments(results, ylim):
        performances = [e.performances[0] for e in results]
        exp_ns = [e.experiment_no for e in results]

        figure = plot_lines(
            performances,
            line_names=[f'Experiment {exp_n}' for exp_n in exp_ns],
            title=f"Performance on Task 1 throughout Experiment(s): {exp_ns}",
            ylabel="Test Accuracy (%) on Task 1",
            xlabel="Batch",
            figsize=(10,5),
            v_line=results[0].switch_indices[:-1],
            v_label='Task switch', ylim=ylim,
            save_as=f"plots/experiments_{exp_ns}_task_1.svg"
        )
    return (plot_task_1_for_all_experiments,)


@app.cell(hide_code=True)
def _(plot_lines):
    def plot_all_tasks_for_experiment(experiment_no, results, ylim):
        experiment = list(filter(lambda x: x.experiment_no == experiment_no, results))[0]
        performances = experiment.performances
        n_batches = experiment.switch_indices[-1]
        x_axes = [range(n_batches - len(p), n_batches) for p in performances]
        task_ns = list(range(1, len(performances) + 1))

        figure = plot_lines(
            performances,
            x_axes=x_axes,
            line_names=[f'Task {task_n}' for task_n in task_ns],
            title=f"Performance on each task throughout Experiment {experiment_no}",
            ylabel="Test Accuracy (%) on Task",
            xlabel="Batch",
            figsize=(10,5),
            v_line=experiment.switch_indices[:-1],
            v_label='Task switch', ylim=ylim,
            save_as=f"plots/experiment_{experiment_no}_all_tasks.svg"
        )
    return (plot_all_tasks_for_experiment,)


@app.cell(hide_code=True)
def _(out_dir):
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
                    axarr.fill_between(x_axes, list(np.array(list_with_lines[line_id]) + np.array(list_with_errors[line_id])),
                                       list(np.array(list_with_lines[line_id]) - np.array(list_with_errors[line_id])),
                                       color=None if (colors is None) else colors[line_id], alpha=0.25)
                else:
                    axarr.plot(x_axes, list(np.array(list_with_lines[line_id]) + np.array(list_with_errors[line_id])), label=None,
                               color=None if (colors is None) else colors[line_id], linewidth=1, linestyle='dashed')
                    axarr.plot(x_axes, list(np.array(list_with_lines[line_id]) - np.array(list_with_errors[line_id])), label=None,
                               color=None if (colors is None) else colors[line_id], linewidth=1, linestyle='dashed')

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
            full_path="{}/{}".format(out_dir, save_as)
            plt.savefig(full_path, bbox_inches='tight')
        if should_show:
            plt.show()


        # return the figure
        return f
    return (plot_lines,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Metrics""")
    return


@app.function
def accuracy_before_task_switches(performance, switch_indices):
    '''
    Compute the accuracy at the last batch before every task switch.
    '''
    res = []
    for i in switch_indices:
        res.append(performance[i - 1])
    return res


@app.function
def gap_depths(performance, switch_indices):
    '''
    Compute the depth of the deepest gap between two consecutive task switches.
    
    The depth is computed as:
    [accuracy before the task switch] - [accuracy at the lowest point at the task switch or after]

    Negative values indicate an increase in accuracy after the task switch.

    The result is an array of tuples with depths and the relative index of the gap (e.g. 0 is the first batch of the new task).
    '''
    accs = accuracy_before_task_switches(performance, switch_indices)
    accs.pop()  # Don't compute for the last task

    res = []
    for i, acc in enumerate(accs):
        start, end = switch_indices[i], switch_indices[i + 1]

        arg_min_per_task = np.argmin(performance[start:end])
        min_per_task = performance[arg_min_per_task]
        
        res.append((acc - min_per_task, arg_min_per_task.item()))
    return res


@app.function
def time_to_recover(performance, switch_indices):
    '''
    Compute time to recover (in number of batches) from the gap to the previous accuracy level.
    Value is computed after every task switch.

        1. Find the lowest gap between two consecutive task switches.
        2. Find the first point after the gap where the accuracy is at least that at the end of the previous task.
        3. Find the difference between time values of the two.

    If the accuracy never recoveres, the resulting value is None.
    '''
    accs = accuracy_before_task_switches(performance, switch_indices)
    accs.pop()  # Don't compute for the last task

    depths = gap_depths(performance, switch_indices)

    res = []

    for i, (_, arg_gap) in enumerate(depths):
        start = switch_indices[i]
        arg_gap_absolute = arg_gap + start
        target = accs[i]
        sliced = np.array(performance[arg_gap_absolute:])
        
        rel_idx = np.argmax(sliced >= target) if np.any(sliced >= target) else None
        if rel_idx == None:
            res.append(None)
        else:
            ttr = (rel_idx + arg_gap).item()
            res.append(ttr)
    
    return res


@app.cell
def _(results):
    print("Accuracy of task 1 before the task switch:")
    for e in results:
        print(f"Experiment {e.experiment_no}:", accuracy_before_task_switches(e.performances[0], e.switch_indices))

    print("\nGap depths of task 1 accuracy:")
    for e in results:
        print(f"Experiment {e.experiment_no}:", gap_depths(e.performances[0], e.switch_indices))

    print("\nTime to recover for task 1 accuracy (in batches):")
    for e in results:
        print(f"Experiment {e.experiment_no}:", time_to_recover(e.performances[0], e.switch_indices))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Utilities""")
    return


@app.cell
def _(displayed):
    def load_results_from_file(results_file, should_log=True):
        results = []
        with open(results_file, 'r') as f:
            if should_log:
                print(f"[INFO] Reading results from {results_file}")
            obj = json.load(f)

        for res_no in displayed:
            res_obj = obj.get(str(res_no), None)
            if res_obj == None:
                continue

            res = ExperimentResult(
                res_obj['experiment_no'],
                res_obj['performances'],
                res_obj['switch_indices'],
            )
            results.append(res)

        results = sorted(results, key=lambda e: e.experiment_no)
        return results
    return (load_results_from_file,)


if __name__ == "__main__":
    app.run()
