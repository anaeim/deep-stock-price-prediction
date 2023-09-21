import argparse


def parse_args():
    """Parse command-line arguments using argparse

    Returns
    -------
    argparse.Namespace
        An object containing parsed command-line arguments
    """

    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg("--ml-model", default='lstm', choices=["lstm","prophet","neuralprophet"], type=str)
    add_arg("--test-size", default=0.35, help="If float, should be between 0.0 and 1.0", type=int)
    add_arg("--dataset", default="AAPL", choices=["AAPL","TSLA"])
    add_arg("--time-stamp", default=100, type=int)
    add_arg("--epoch", default=54)
    add_arg("--batch-size", default=84)
    add_arg("--enable-save-model", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    print(parse_args())