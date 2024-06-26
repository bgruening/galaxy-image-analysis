import argparse
import warnings

import numpy as np
import pandas as pd
import skimage.io


def binaryimage2points(input_file):
    # ignore warnings that arise when importing a package that was compiled against an older version of numpy than installed; https://github.com/numpy/numpy/pull/432
    warnings.filterwarnings("ignore")

    img_in = skimage.io.imread(input_file, plugin='tifffile')

    # make label image
    label = skimage.measure.label(img_in)

    # amount of regions
    amount_label = np.max(label)

    # iterate over all regions in order to calc center of mass
    center_mass = []
    for i in range(1, amount_label + 1):
        # get coordinates of region
        coord = np.where(label == i)
        # be carefull with x,y coordinates
        center_mass.append([np.mean(coord[1]), np.mean(coord[0])])

    # make data frame of detections
    out_dataFrame = pd.DataFrame(center_mass)

    # return
    return out_dataFrame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input file')
    parser.add_argument('out_file', help='out file (TSV)')

    args = parser.parse_args()
    input_file = args.input_file
    out_file = args.out_file

    # TOOL
    out_dataFrame = binaryimage2points(input_file)

    # Print to csv file
    out_dataFrame.to_csv(out_file, index=False, header=False, sep="\t")
