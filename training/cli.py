import argparse


def parse_args():
	"""
	Parse sys.argv arguments for the training program.
	"""
	parser = argparse.ArgumentParser(prog="poetry run python -m training.main")

	parser.add_argument("--seed", "-s", type=int, help="seed to initialize numpy and torch with", default=None)

	return parser.parse_args()
