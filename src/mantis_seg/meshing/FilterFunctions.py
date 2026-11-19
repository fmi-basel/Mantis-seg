# Copyright (C) 2024 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from mantis_seg.linking.NucleiLinkingFunctions import calculate_quantiles


def filter_small_sizes_per_round(props_numpy, column=-1, threshold=0.05):
    """
    props_np: numpy array, where each row corresponds to a nucleus and columns are regionprops measurements
    Filter out nuclei with small volumes, presumed to be debris.
    Column: index of column in numpy array that correspond to volume measurement
    Threshold: multiplier that specifies cutoff for volumes below which nuclei are filtered out, float in range [0,1],
        e.g. 0.05 means that 5% of median of nuclear volume distribution is used as cutoff.
    Return filtered numpy array (moving_pc_filtered, fixed_pc_filtered) as well as logging metrics.
    """
    median, _ = calculate_quantiles(props_numpy, q=0.5, column=column)

    # generate boolean arrays for filtering
    volume_cutoff = threshold * median
    row_selection = props_numpy[:, column].transpose() > volume_cutoff

    # filter pc to remove small nuclei
    props_numpy_filtered = props_numpy[row_selection, :]

    # return nuclear ids of removed nuclei, assume nuclear labels are first column (column 0)
    removed = props_numpy[~row_selection, 0]

    # calculate mean of removed nuclear sizes
    if props_numpy[~row_selection, column].size == 0:
        removed_size_mean = None
    else:
        removed_size_mean = np.mean(props_numpy[~row_selection, column])

    # calculate mean of kept nuclear sizes
    size_mean = np.mean(props_numpy_filtered[:, column])

    return props_numpy_filtered, removed, removed_size_mean, size_mean, volume_cutoff


def calculate_mean_volume(seg):
    """
    Calculate mean volume across labels in input numpy array segmentation image
    """
    props = regionprops_table(label_image=seg, properties=("label", "area"))

    # output column order must be: ["label", "volume"]
    # units are in pixels
    props = (pd.DataFrame(props, columns=["label", "area"])).to_numpy()

    # calculate mean of nuclear volumes
    size_mean = np.mean(props[:, -1])

    return size_mean


def load_border_values(nparray):
    """
    Load edge (border) values of a 2-dim numpy array
    Return: 1-d numpy array of edge values
    """
    if len(nparray.shape) != 2:
        raise ValueError("Expecting 2-dimensional image (z-slice) as input")
    alist = [nparray[0, :], nparray[1:, -1], nparray[-1, :-1], nparray[1:-1, 0]]
    edge_values = np.concatenate(alist)
    return edge_values


def remove_border(nparray, pad_width=1):
    """
    Remove edge (border) values from a 2-dim numpy array, e.g. to reverse padding
    When n defines width of pad to remove (integer)
    Return: 2-d numpy array with edge values removed
    """
    if len(nparray.shape) != 2:
        raise ValueError("Expecting 2-dimensional image (z-slice) as input")
    return nparray[pad_width:-pad_width, pad_width:-pad_width]

def min_nonzero_label(label_img):
    """
    Find lowest label in segmentation image that is not 0 background label
    """
    nonzero_vals = label_img[label_img > 0]
    return np.min(nonzero_vals) if nonzero_vals.size > 0 else None