import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection import *
from daneel.detection.transit import *
from daneel.detection.detection_methods import *
from daneel.dream.dream import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        required=False,
        help="Initialise detection algorithms for Exoplanets",
        action="store",
    )

    parser.add_argument(
        "-a",
        "--atmosphere",
        dest="atmosphere",
        required=False,
        help="Atmospheric Characterisazion from input transmission spectrum",
        action="store_true",
    )

    parser.add_argument(
        "-t",
        "--transit",
        dest="transit",
        required=False,
        help="If present compute the light curve",
        action="store_true"
    )
    
    parser.add_argument(
        "--dream",
        dest="dream",
        required=False,
        help="Dream a new world",
        action="store_true",
    )

    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")
    param = Parameters(args.input_file)
    input_pars = param.params
    if args.transit:
        calculate_transit(param)
    if args.detect == "svm":
        DetectionWithSVM(param)
    if args.detect == "nn":
        DetectionWithNN(param)
    if args.detect == "cnn":
        DetectionWithCNN(param)
    
    if args.atmosphere:
        pass
    
    if args.dream:
        DreamNewWorlds(param)

    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")


if __name__ == "__main__":
    main()
