import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np

from .metrics import average_accuracy

PLOT_CONFIG = {
	"linewidth_thin": 1.75,
	"linewidth_thick": 3.15,
	"palette1": [(0.956, 0.878, 1.0),(0.551, 1.0, 0.65), (0.385, 0.699, 0.639)],  # HSV, categorical
	"palette2": [(0.772, 0.477, 0.765), (0.324, 0.356, 0.789)],  # HSV, categorical
	"palette3": ["#414c66", "#0000b3", "#0020ff", "#0080ff", "#00beee"],  # For sequential data
	"palette4": ["#ef9b20", "#27aeef", "#87bc45"], # Categorical
	"xticks_size": 16,
	"yticks_size": 22,
	"title_size": 22,
	"label_size": 18,
	"show_title": False,
	"legend_size": 16
}


def setup():
	plt.style.use('default')
	plt.rcParams['lines.linewidth'] = PLOT_CONFIG["linewidth_thin"]
	plt.rcParams['xtick.labelsize'] = PLOT_CONFIG["xticks_size"]
	plt.rcParams['ytick.labelsize'] = PLOT_CONFIG["yticks_size"]
	plt.rcParams['axes.titlesize'] = PLOT_CONFIG["title_size"]
	plt.rcParams['axes.labelsize'] = PLOT_CONFIG["label_size"]
	plt.rcParams['legend.fontsize'] = PLOT_CONFIG["legend_size"]
	plt.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"text.latex.preamble": r"\usepackage{times}",
	})


def generate_plots(results, plots_dir, format, should_show=False):
	line_names = [
		"Baseline",
		"ND"
	]
	y_lim_zoomed = (75, 100)
	y_lim_unzoomed = (50, 100)
	show_std = True

	results.sort(key=lambda e: e.experiment_no)

	# Task 1, standard parameters
	shrunk = list(filter(lambda e: e.experiment_no == 1 or e.experiment_no == 4, results))
	plot_task_1_for_all_experiments(shrunk, show_std, y_lim_zoomed, plots_dir, format, should_show, line_names)

	# Average, standard parameters
	shrunk = list(filter(lambda e: e.experiment_no == 1 or e.experiment_no == 4, results))
	plot_average_accuracy(shrunk, y_lim_unzoomed, plots_dir, should_show, line_names)

	# Task 1, various lambdas
	shrunk = list(filter(lambda e: e.experiment_no == 1 or e.experiment_no == 4 or e.experiment_no == 5 or e.experiment_no == 6, results))
	line_names_override = [  # Hardcoded values (because parsing them is painful)
		"Baseline",
		"$\\lambda = 1 \\cdot 10^{-7}$",
		"$\\lambda = 1 \\cdot 10^{-5}$",
		"$\\lambda = 5 \\cdot 10^{-5}$"]
	shrunk[1], shrunk[2] = shrunk[2], shrunk[1]  # Experiment 4 should be in the middle
	plot_task_1_for_all_experiments(shrunk, False, y_lim_zoomed, plots_dir, format, should_show, line_names_override)

	# Task 1, lower learning rate
	shrunk = list(filter(lambda e: e.experiment_no == 7 or e.experiment_no == 8, results))
	plot_task_1_for_all_experiments(shrunk, show_std, y_lim_zoomed, plots_dir, format, should_show, line_names)

	# Task 1, higher learning rate
	shrunk = list(filter(lambda e: e.experiment_no == 9 or e.experiment_no == 10, results))
	plot_task_1_for_all_experiments(shrunk, show_std, y_lim_zoomed, plots_dir, format, should_show, line_names)

	# Task 1, smaller net
	shrunk = list(filter(lambda e: e.experiment_no == 11 or e.experiment_no == 12, results))
	plot_task_1_for_all_experiments(shrunk, show_std, y_lim_zoomed, plots_dir, format, should_show, line_names)

	# Task 1, bigger net
	shrunk = list(filter(lambda e: e.experiment_no == 13 or e.experiment_no == 14, results))
	plot_task_1_for_all_experiments(shrunk, show_std, y_lim_zoomed, plots_dir, format, should_show, line_names)

	# All tasks, baseline, standard parameters
	plot_all_tasks_for_experiment(1, results, show_std, y_lim_zoomed, plots_dir, should_show)

	# All tasks, experimental, standard parameters
	plot_all_tasks_for_experiment(4, results, show_std, y_lim_zoomed, plots_dir, should_show)


def plot_task_1_for_all_experiments(results, show_std, ylim, plots_dir, format, should_show, line_names):
	setup()
	performances = [e.performances[0] for e in results]
	stds = None
	if show_std:
		try:
			stds = [e.stds[0] for e in results]
		except TypeError:
			stds = None
	exp_ns = [e.experiment_no for e in results]

	colors = [hsv_to_rgb(c) for c in PLOT_CONFIG["palette1"]]
	if len(performances) >= 4:
		colors = PLOT_CONFIG["palette3"]

	figure = plot_lines(
		performances,
		list_with_errors=stds,
		line_names=line_names,
		title=f"Performance in Task 1 with and without neuronal decay" if PLOT_CONFIG["show_title"] else None,
		ylabel="Test accuracy in Task 1 (\\%)",
		xlabel="Batch",
		figsize=(10,5),
		v_line=results[0].switch_indices[:-1],
		v_label='Task switch',
		ylim=ylim,
		colors=colors,
		save_as=f"{plots_dir}/experiments_{exp_ns}_task_1.{format}",
		should_show=should_show,
	)


def plot_all_tasks_for_experiment(experiment_no, results, show_std, ylim, plots_dir, should_show):
	setup()
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

	colors = PLOT_CONFIG["palette4"]

	figure = plot_lines(
		performances,
		x_axes=x_axes,
		list_with_errors=stds,
		line_names=[f'Task {task_n}' for task_n in task_ns],
		title=f"Performance in each task throughout Experiment {experiment_no}" if PLOT_CONFIG["show_title"] else None,
		ylabel="Test accuracy in the given task (\\%)",
		xlabel="Batch",
		figsize=(10,5),
		v_line=experiment.switch_indices[:-1],
		v_label='Task switch',
		ylim=ylim,
		colors=colors,
		save_as=f"{plots_dir}/experiment_{experiment_no}_all_tasks.pdf",
		should_show=should_show,
	)


def plot_average_accuracy(results, ylim, plots_dir, should_show, line_names):
	setup()
	performances = [e.performances for e in results]
	exp_ns = [e.experiment_no for e in results]
	switch_indices = results[0].switch_indices

	avg_accuracies = [average_accuracy(perfs, switch_indices) for perfs in performances]

	colors = [hsv_to_rgb(c) for c in PLOT_CONFIG["palette2"]]
	lw = PLOT_CONFIG["linewidth_thick"]

	figure = plot_lines(
		avg_accuracies,
		line_names=line_names,
		title=f"Average accuracy in all tasks with and without decay" if PLOT_CONFIG["show_title"] else None,
		ylabel="Average test accuracy in all tasks (\\%)",
		xlabel="Batch",
		figsize=(10,5),
		v_line=switch_indices[:-1],
		v_label='Task switch',
		ylim=ylim,
		lw=lw,
		colors=colors,
		with_zoom=False,
		save_as=f"{plots_dir}/experiments_{exp_ns}_avg.pdf",
		should_show=should_show,
	)


# Code taken, with some adjustments, from:
# https://github.com/GMvandeVen/continual-learning/blob/50b8b7fce9786dc402866fc8387e1525f369bbc5/visual/visual_plt.py#L103
def plot_lines(list_with_lines, x_axes=None, line_names=None, colors=None, title=None,
			   title_top=None, xlabel=None, ylabel=None, ylim=None, figsize=None, list_with_errors=None, errors="shaded",
			   x_log=False, with_dots=False, linestyle='solid', lw=None, h_line=None, h_label=None, h_error=None,
			   h_lines=None, h_colors=None, h_labels=None, h_errors=None, with_zoom=True,
			   v_line=None, v_label=None, save_as=None, should_show=True, should_log=True):
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

	x_ticks = [0] + v_line + [len(list_with_lines[0])]
	axarr.set_xticks(x_ticks)

	# Picture in picture
	if with_zoom:
		axins_arr = picture_in_picture(f, axarr, list_with_lines, ylim, v_line)
	else:
		axins_arr = []

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
				for axins in axins_arr:
					axins.fill_between(
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
				   linewidth=lw, marker='o' if with_dots else None, linestyle=linestyle if type(linestyle)==str else linestyle[line_id])
		for axins in axins_arr:
			axins.plot(x_axes[line_id], list_with_lines[line_id], label=name,
					   color=None if (colors is None) else colors[line_id],
					   linewidth=lw, marker='o' if with_dots else None, linestyle=linestyle if type(linestyle)==str else linestyle[line_id])

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
		ls = ":"
		zorder = 0  # Show below main lines
		cl = (0, 0, 0, 0.25)
		lw = 3
		if type(v_line)==list:
			for id,new_line in enumerate(v_line):
				axarr.axvline(x=new_line, ls=ls, label=v_label if id==0 else None, color=cl, linewidth=lw, zorder=zorder)
		else:
			axarr.axvline(x=v_line, ls=ls, label=v_label, color=cl, linewidth=lw, zorder=zorder)

	axarr.spines[['right', 'top']].set_visible(False)


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
		legend = axarr.legend(loc='lower right')
		legend.get_frame().set_facecolor((0.92, 0.92, 0.92))
	# -set x-axis to log-scale
	if x_log:
		axarr.set_xscale('log')
	if save_as is not None:
		if should_log:
			print(f"[INFO] Saving plot to {save_as}")
		plt.savefig(save_as, bbox_inches='tight')
	if should_show:
		plt.show()

	plt.close()

	# return the figure
	return f


def picture_in_picture(f, axarr, list_with_lines, ylim, v_line):
	axins_arr = []
	for idx, v in enumerate(v_line):
		x1, x2 = v - 10, v + 100
		y1, y2 = 80, 100

		total_w = len(list_with_lines[0])
		total_h = ylim[1] - ylim[0]

		# This puts the inner axis roughly at the same spot
		inset_w = (x2 - x1) / total_w
		inset_h = (y2 - y1) / total_h
		inset_left = x1 / total_w
		inset_bottom = (y1 - ylim[0]) / total_h

		# Scale
		inset_w *= 4
		inset_h *= 0.4

		# Move
		if idx == 0:
			inset_left *= 0.53
			inset_bottom += 0.88
		elif idx == 1:
			inset_left *= 0.86
			inset_bottom += 0.88
		else:
			NotImplementedError("More than two switch indexes: manual inset axes positioning required")

		bbox = (inset_left, inset_bottom, inset_w, inset_h)
		inner_border = "gray"
		aoi_border = "gray"

		axins = inset_axes(axarr, width="100%", height="100%", loc='lower left',
						   bbox_to_anchor=bbox, bbox_transform=axarr.transAxes)

		for spine in axins.spines.values():
			spine.set_edgecolor(inner_border)
		axins.spines[['right', 'top']].set_visible(False)

		axins.set_xlim(x1, x2)
		axins.set_ylim(y1, y2)
		axins.set_xticks([x1, x2])
		axins.set_yticks([y1, y2])

		axins.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

		# axins.grid(zorder=100)

		# Connect with dashed lines
		axarr.annotate(
			'',
			xy=(x1, y2), xycoords=axarr.transData,
			xytext=(0, 0), textcoords=axins.transAxes,
			arrowprops=dict(arrowstyle='-', linestyle=':', color=aoi_border, lw=1,
							shrinkA=0, shrinkB=0)
		)
		axarr.annotate(
			'',
			xy=(x2, y2), xycoords=axarr.transData,
			xytext=(1, 0), textcoords=axins.transAxes,
			arrowprops=dict(arrowstyle='-', linestyle=':', color=aoi_border, lw=1,
							shrinkA=0, shrinkB=0)
		)

		axins.set_facecolor((0.97, 0.97, 0.96))

		axins.tick_params(axis='x', labelsize=16)
		axins.tick_params(axis='y', labelsize=16)

		_, c1, c2 = mark_inset(axarr, axins, loc1=1, loc2=2, ec=aoi_border, lw=1.2, ls=":", zorder=20)
		c1.set_visible(False)
		c2.set_visible(False)

		axins_arr.append(axins)

	return axins_arr