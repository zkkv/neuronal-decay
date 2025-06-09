from .rotations import plot_rotations


def main():
	format = "pdf"
	plot_rotations(save_as=f"./out/misc/rotations.{format}")


if __name__ == "__main__":
	main()
