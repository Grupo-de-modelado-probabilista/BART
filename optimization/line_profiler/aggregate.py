import argparse


def main(args):

    for file in args.files:
        with open(file, "r") as f:
            for line in f:
                data = line.split()
                print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", help="Files to aggregate")
    args = parser.parse_args()
    main(args)
    