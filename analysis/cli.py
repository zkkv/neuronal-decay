import argparse


def parse_args():
	"""
	Parse sys.argv arguments for the analysis program.
	"""
	parser = argparse.ArgumentParser(prog="poetry run python -m analysis.main")

	parser.add_argument("seeds", nargs="*", type=int,
						help="zero or more seeds to use when reading result files")
	parser.add_argument("--display", "-d", nargs="+", type=int, default=[],
						help="one or more experiment numbers to be displayed in the plots, default is all")

	return parser.parse_args()
