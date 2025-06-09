from .rotations import plot_rotations
from .stability_gap import plot_stability_gap
from utilities.fs import make_dirs
from utilities.meta import MISC_DIR


def main():
	make_dirs([MISC_DIR])
	format = "pdf"
	plot_rotations(save_as=f"{MISC_DIR}/rotations.{format}")
	plot_stability_gap(save_as=f"{MISC_DIR}/stability_gap.{format}")


if __name__ == "__main__":
	main()
