import matplotlib.pyplot as plt

from training.data import get_datasets
from training.config import Params
from utilities.meta import DATA_DIR


def setup():
	plt.style.use('default')
	plt.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"text.latex.preamble": r"\usepackage{times}\usepackage{gensymb}\usepackage{amssymb}",
	})


def plot_rotations(save_as):
	setup()
	rotations = Params.rotations
	sets, _ = get_datasets(DATA_DIR, rotations, limit=10)

	idx_datapoint = 7

	rotated = []
	for i in range(len(rotations)):
		x = sets[i][idx_datapoint][0]  # Take the raw value
		rotated.append(x)

	h, w = 1, len(rotated)
	fig, axes = plt.subplots(h, w, figsize=(6, 6))

	for task, img in enumerate(rotated):
		ax = axes[task]
		ax.imshow(img.permute(1, 2, 0), cmap='gray')
		part_1 = f"Task {task + 1}"
		part_2 = f"${rotations[task]}\\degree \\curvearrowleft$"
		ax.text(0.00, 1.05, part_1, transform=ax.transAxes,
				ha='left', va='bottom', fontsize=18, weight='regular')
		ax.text(0.98, 1.05, part_2, transform=ax.transAxes,
				ha='right', va='bottom', fontsize=18, weight='regular')
		ax.axis('off')

	plt.tight_layout()
	plt.savefig(save_as, bbox_inches='tight')
	plt.close()
