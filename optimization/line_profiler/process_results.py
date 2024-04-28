import argparse

def main(args):

    print(f"Processing: {args.file}")

    for file in args.file:
        with open(file, "r") as file:
            lines = file.readlines()

    for line in lines:
        # if "function" in line:
        print(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results from line_profiler")
    parser.add_argument("--file", nargs="*")
    args = parser.parse_args()
    main(args)