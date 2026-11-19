# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#
#                                                                            #
##############################################################################

from copy import deepcopy

import dask.array as da
import numpy as np
import pandas as pd


def calculate_quantiles(pc, q=0.5, column=-1):
    """
    Calculate mean and standard deviation of nuclear volumes in organoid
    Helper function used during affine linking
    q is quantile in range [0,1] to calclulate
    """
    quantile = np.quantile(pc[:, column].transpose(), q)
    iqr = np.quantile(pc[:, column].transpose(), 0.75) - np.quantile(
        pc[:, column].transpose(), 0.25
    )
    return quantile, iqr


def make_linking_dict(
    matches: pd.DataFrame,
    moving_colname: str = "RX_nuc_id",
    fixed_colname: str = "R0_nuc_id",
) -> dict:
    """
    Create a dictionary mapping moving labels to fixed labels.

    Parameters
    ----------
    matches : pandas.DataFrame
        DataFrame containing at least the moving and fixed label columns.
    moving_colname : str
        Column name representing the original (moving) labels.
    fixed_colname : str
        Column name representing the target (fixed) labels.

    Returns
    -------
    dict
        Dictionary mapping {moving_label -> fixed_label}.

    Notes
    -----
    If duplicate moving labels exist, later values may overwrite earlier ones.
    """
    linking_dict = matches.set_index(moving_colname).T.to_dict("index")[fixed_colname]
    return linking_dict


def make_relabeled_block(
    block1: np.ndarray,
    matching_dict: dict,
    label_dtype: np.dtype,
) -> np.ndarray:
    """
    Relabel a NumPy array block using a vectorized lookup table (LUT).

    This function remaps integer label values in `block1` according to
    `matching_dict`, returning a new array of the same shape with updated
    labels. Labels not present in the mapping are set to zero. The operation
    is fully vectorized using NumPy and avoids Python-level loops over pixels,
    making it efficient for large array chunks (e.g., in Dask blockwise
    processing).

    Parameters
    ----------
    block1 : numpy.ndarray
        Input labeled array block. Expected to contain non-negative integer
        label values, where 0 typically represents background.
    matching_dict : dict
        Dictionary mapping original label IDs (keys) to new label IDs (values).
        Keys and values should be integers. Labels not found in this dictionary
        will be mapped to 0 in the output.
    label_dtype : numpy.dtype
        Desired dtype of the output array (e.g., `np.uint32`), used both for
        the lookup table and the returned array.

    Returns
    -------
    numpy.ndarray
        A new NumPy array with the same shape as `block1`, where label values
        have been replaced according to `matching_dict` and cast to
        `label_dtype`.

    Notes
    -----
    - A lookup table (LUT) of size `(max_label + 1)` is constructed, where
      `max_label` is the largest label value present in `block1`.
    - Each index in the LUT corresponds to an original label ID; the value
      stored at that index is the new label ID.
    - The relabeling is performed in a single vectorized operation:
      `lut[block1]`, which is significantly faster than iterating over pixels.
    - Memory usage scales with the maximum label value in `block1`. If label
      IDs are extremely large or sparse, consider using a sparse or hashed
      remapping strategy instead.

    Examples
    --------
    >> block = np.array([[0, 1, 2], [2, 5, 0]])
    >> mapping = {1: 10, 2: 20, 5: 99}
    >> make_relabeled_block(block, mapping, np.uint32)
    array([[ 0, 10, 20],
           [20, 99,  0]], dtype=uint32)
    """
    # Build LUT (lookup table)
    # Find largest label in block
    max_label = block1.max()
    # Initialize empty look up table (all zeros). Index is old label; value at index is new label.
    lut = np.zeros(max_label + 1, dtype=label_dtype)

    # Fill LUT using mapping dictionary
    for k, v in matching_dict.items():
        k_int = int(k)
        v_int = int(v)
        if k_int <= max_label:
            lut[k_int] = v_int

    # Vectorized remap
    return lut[block1]


def run_relabel_dask(
    label_dask: da.Array,
    matches: pd.DataFrame = None,
    moving_colname: str = "RX_nuc_id",
    fixed_colname: str = "R0_nuc_id",
    matching_dict: dict = None,
) -> da.Array:
    """
    Apply label remapping to a Dask array using block-wise relabeling.

    Parameters
    ----------
    label_dask : dask.array.Array
        Input labeled Dask array to relabel (moving label).
    matches : pandas.DataFrame, optional
        DataFrame mapping moving labels to fixed labels. Required if `matching_dict` is not provided.
    moving_colname : str, default="RX_nuc_id"
        Column containing original label IDs (only used if `matches` is provided).
    fixed_colname : str, default="R0_nuc_id"
        Column containing target label IDs (only used if `matches` is provided).
    matching_dict : dict, optional
        Precomputed mapping dictionary {moving_label -> fixed_label}, typically integer values.
        If provided, `matches` and column names are ignored.

    Returns
    -------
    dask.array.Array
        Lazily evaluated Dask array with relabeled values.

    Notes
    -----
    - Uses `da.map_blocks` to process chunks independently.
    - Output dtype matches input dtype.
    - Labels not found in the mapping are set to zero.
    - Either `matching_dict` must be provided, or `matches` DataFrame with the specified columns.

    """

    if matching_dict is None:
        if matches is None:
            raise ValueError("Either 'matches' or 'matching_dict' must be provided.")
        # key is moving_label (current label), value is fixed_label (value to rename to)
        matching_dict = make_linking_dict(matches, moving_colname, fixed_colname)

    dtype = label_dask.dtype

    relabeled_dask = da.map_blocks(
        make_relabeled_block,
        label_dask,
        matching_dict=matching_dict,
        label_dtype=dtype,
        dtype=dtype,
    )

    return relabeled_dask


def remove_labels(seg_img, labels_to_remove, datatype):
    """
    Remove labels from segmentation image.
    labels_to_remove is list of labels, and each value should match data format of
    """
    seg_img_relabeled = deepcopy(seg_img)

    for lab in labels_to_remove:
        lab = lab.astype(datatype)
        seg_img_relabeled[seg_img_relabeled == lab] = 0

    return seg_img_relabeled