#!/usr/bin/env python3
# This script allows you to test that all necessary dependencies are installed

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

	print("All dependencies were found!")


if __name__ == "__main__":
	test()