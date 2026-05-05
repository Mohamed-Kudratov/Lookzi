import argparse
from IPython.display import HTML, display


def parse_args():
    parser = argparse.ArgumentParser(description="Display a Lookzi benchmark HTML report inside Colab.")
    parser.add_argument("report_path", help="Path to review_report.html")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.report_path, "r", encoding="utf-8") as f:
        display(HTML(f.read()))


if __name__ == "__main__":
    main()
