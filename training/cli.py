import argparse


def parse_args():
	"""
	Parse sys.argv arguments for the training program.
	"""
	parser = argparse.ArgumentParser(prog="poetry run python -m training.main")

	parser.add_argument("--seed", "-s", type=int, default=None,
						help="seed to initialize numpy and torch with")
	parser.add_argument("--lambda", "-l", dest="lam", metavar="LAMBDA", type=float, default=None,
						help="neuronal decay lambda")
	parser.add_argument("--quiet", "-q", action="store_true", default=False,
						help="suppress all output to stdout")
	parser.add_argument("--no-log", "-n", action="store_true", default=False,
						help="suppress logging to a file")
	parser.add_argument("--profile", "-p", action="store_true", default=False,
						help="profile the training step")

	return parser.parse_args()
