#!/usr/bin/env python3
# This script allows you to confirm that all required dependencies are found. Also see: import_test.sh.

def test():
	# Build-ins
	import json
	import os
	import dataclasses
	import argparse

	# External
	import torch
	import torchvision
	import matplotlib
	import numpy
	import thop

	pkgs = [
		(torch.__name__, torch.__version__),
		(torchvision.__name__, torchvision.__version__),
		(matplotlib.__name__, matplotlib.__version__),
		(numpy.__name__, numpy.__version__),
		(thop.__name__, thop.__version__),
	]
	for n, v in pkgs:
		print(f"{n:<20} {v:<20}")
	print("All dependencies were found!")


if __name__ == "__main__":
	test()