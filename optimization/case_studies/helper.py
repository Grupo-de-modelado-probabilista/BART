import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=50, help="Number of trees")
    parser.add_argument("--particle", type=int, default=20, help="Number of particles")
    parser.add_argument("--cores", type=int, default=4, help="Number of cores")
    return parser.parse_args()
