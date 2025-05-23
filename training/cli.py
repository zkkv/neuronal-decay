import argparse


def parse_args():
	"""
	Parse sys.argv arguments for the training program.
	"""
	parser = argparse.ArgumentParser(prog="poetry run python -m training.main")

	parser.add_argument("--seed", "-s", type=int, help="seed to initialize numpy and torch with", default=None)
	parser.add_argument("--lambda", "-l", dest="lam", metavar="LAMBDA", type=float, help="neuronal decay lambda", default=None)

	return parser.parse_args()
