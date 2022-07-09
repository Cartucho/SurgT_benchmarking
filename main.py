import argparse
from src.utils import load_yaml_data, download_data
from src.evaluate import evaluate_method


def main():
    parser = argparse.ArgumentParser(description='Tool to label stereo matches')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('-nv', '--no-visualization', help="no visualization is shown.", action="store_true")
    args = parser.parse_args()
    config = load_yaml_data(args.config)
    # Download data (if not downloaded before)
    download_data(config)
    # Run method
    evaluate_method(config, args.no_visualization)


if __name__ == "__main__":
    main()
