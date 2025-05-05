

import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import json
    from src.training import ExperimentResult


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Evaluation""")
    return


@app.cell
def _():
    out_dir = "./out"
    result_dir = f"{out_dir}/results"

    print(f"[INFO] Result directory: {result_dir}")
    return out_dir, result_dir


@app.cell
def _(result_dir):
    results = load_results_from_file(result_dir)
    return (results,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Visualization""")
    return


@app.cell
def _(plot_all, results):
    plt.style.use('default')

    plot_all(results, ylim=(0, 100))
    return


@app.cell(hide_code=True)
def _(plot_lines):
    def plot_individually(results, ylim):
        for result in results:
            performances = [result.performance]
            exp_n = result.experiment_no

            figure = plot_lines(
                performances,
                line_names=['Performance'],
                title=f"Performance on Task 1 throughout Experiment {exp_n}",
                ylabel="Test Accuracy (%) on Task 1",
                xlabel="Batch",
                figsize=(10,5),
                v_line=result.switch_indices[:-1],
                v_label='Task switch', ylim=ylim,
                save_as=f"plots/experiment_{exp_n}.svg"
            )
    return


@app.cell(hide_code=True)
def _(plot_lines):
    def plot_all(results, ylim):
        performances = [e.performance for e in results]
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
            save_as=f"plots/experiment_{exp_ns}.svg"
        )
    return (plot_all,)


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
            x_axes = list(range(n_obs))

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
            axarr.plot(x_axes, list_with_lines[line_id], label=name,
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
def accuracy_at_task_switches(performance, switch_indices):
    res = []
    for i in switch_indices:
        res.append(performance[i - 1])
    return res


@app.function
def gap_depths(performance, switch_indices):
    accs = accuracy_at_task_switches(performance, switch_indices)
    accs.pop()

    res = []
    for i, acc in enumerate(accs):
        start, end = switch_indices[i], switch_indices[i + 1]
        min_per_task = min(performance[start:end])
        res.append(acc - min_per_task)
    return res


@app.cell
def _(results):
    print("Accuracy at task switches:")
    for e in results:
        print(accuracy_at_task_switches(e.performance, e.switch_indices))

    print("\nGap depths:")
    for e in results:
        print(gap_depths(e.performance, e.switch_indices))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Utilities""")
    return


@app.function
def load_results_from_file(dir):
    results = []
    for path in os.listdir(dir):
        if not path.endswith(".json"):
            continue

        with open(f"{dir}/{path}", 'r') as f:
            obj = json.load(f)
        
        res = ExperimentResult(
            obj['experiment_no'],
            obj['performance'],
            obj['switch_indices'],
        )
        results.append(res)

    results = sorted(results, key=lambda e: e.experiment_no)
    return results


if __name__ == "__main__":
    app.run()
