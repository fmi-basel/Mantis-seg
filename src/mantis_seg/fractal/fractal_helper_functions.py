
# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Joel Lüthi <joel.luethi@uzh.ch>
#

import logging
import os
from typing import Tuple

import anndata as ad
import dask.array as da
import ngio
import numpy as np
import pandas as pd
import zarr
from fractal_tasks_core.channels import OmeroChannel, get_channel_from_image_zarr
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    array_to_bounding_box_table,
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    empty_bounding_box_table,
    load_region,
)
from zarr.errors import ArrayNotFoundError

logger = logging.getLogger(__name__)


def get_zattrs(zarr_url):
    with zarr.open(zarr_url, mode="r") as zarr_img:
        return zarr_img.attrs.asdict()


def convert_indices_to_origin_zyx(
    index: list[int],
) -> tuple[int, int, int]:
    """
    Converts index tuples to origin zyx tuple

    Args:
        index: Tuple containing 6 entries of (z_start, z_end, y_start,
            y_end, x_start, x_end).

    Returns:
        region: tuple of 3 integers (z_start, y_start, x_start)
    """
    return index[0], index[2], index[4]


def format_roi_table(bbox_dataframe_list):
    """
    Formats ROI table to anndata
    Copied from cellpose Fractal task
    Returns anndata to save
    """
    # Remove empty dataframes
    bbox_dataframe_list = [df for df in bbox_dataframe_list if not df.empty]
    # Handle the case where `bbox_dataframe_list` is empty (typically
    # because list_indices is also empty)
    if len(bbox_dataframe_list) == 0:
        bbox_dataframe_list = [empty_bounding_box_table()]
    # Concatenate all ROI dataframes
    df_well = pd.concat(bbox_dataframe_list, axis=0, ignore_index=True)
    df_well.index = df_well.index.astype(str)
    # Extract labels and drop them from df_well
    labels = pd.DataFrame(df_well["label"].astype(str))
    df_well.drop(labels=["label"], axis=1, inplace=True)
    # Convert all to float (warning: some would be int, in principle)
    bbox_dtype = np.float32
    df_well = df_well.astype(bbox_dtype)
    # Convert to anndata
    bbox_table = ad.AnnData(df_well)
    bbox_table.obs = labels

    return bbox_table


def initialize_new_label(
    zarr_url,
    shape,
    chunks,
    label_dtype,
    inherit_from_label,
    output_label_name,
    logger,
):
    store = zarr.storage.FSStore(f"{zarr_url}/labels/{output_label_name}/0")

    if len(shape) != 3 or len(chunks) != 3:
        raise ValueError(
            "Array must be 3D; if input image is 2D, the 3rd dimension must be set to 1. "
        )

    if shape[0] == 1:
        logger.info("Initializing 2D label image array.")
    else:
        logger.info("Initializing 3D label image array.")

    # Add metadata to labels group
    # Get the label_attrs correctly
    # Note that the new label metadata matches the label_name (child) metadata
    label_attrs = get_zattrs(zarr_url=f"{zarr_url}/labels/{inherit_from_label}")
    _ = prepare_label_group(
        image_group=zarr.group(zarr_url),
        label_name=output_label_name,
        overwrite=True,
        label_attrs=label_attrs,
        logger=logger,
    )

    new_label3d_array = zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=True,
        dimension_separator="/",
    )
    return new_label3d_array


def save_new_label_with_overlap(
    new_npimg,
    new_label3d_array,
    zarr_url,
    output_label_name,
    region,
    compute,
):
    # Load dask from disk, will contain rois of the previously processed objects within for loop
    new_label3d_dask = da.from_zarr(f"{zarr_url}/labels/{output_label_name}/0")
    # Load region of current object from disk, will include any previously processed neighboring objects
    seg_ondisk = load_region(
        data_zyx=new_label3d_dask,
        region=region,
        compute=compute,
    )

    # Check that dimensions of rois match
    if seg_ondisk.shape != new_npimg.shape:
        raise ValueError(
            "Computed label image must match image dimensions of bounding box during saving"
        )

    # Use fmax so that if one of the elements being compared is a NaN, then the non-nan element is returned
    new_npimg_label_tosave = np.fmax(new_npimg, seg_ondisk)

    # Compute and store 0-th level of new 3d label map to disk
    da.array(new_npimg_label_tosave).to_zarr(
        url=new_label3d_array,
        region=region,
        compute=True,
    )
    return


def save_new_label_and_bbox_df(
    new_npimg,
    new_label3d_array,
    zarr_url,
    output_label_name,
    region,
    label_pixmeta,
    compute,
    roi_idlist,
    row_int,
):
    save_new_label_with_overlap(
        new_npimg, new_label3d_array, zarr_url, output_label_name, region, compute
    )

    # make new ROI table
    origin_zyx = convert_indices_to_origin_zyx(roi_idlist[row_int])

    bbox_df = array_to_bounding_box_table(
        new_npimg,
        label_pixmeta,
        origin_zyx=origin_zyx,
    )

    return bbox_df


def roi_to_pixel_slices(
    roi: ngio.Roi, spacing: np.ndarray
) -> Tuple[slice, slice, slice]:
    """
    Convert Roi (with x,y,z and lengths in um) to pixel slices.

    Args:
        roi: object with attributes roi.x, roi.y, roi.z, roi.x_length, roi.y_length, roi.z_length (all in um)
        spacing: numpy array of shape (3,), with elements [um/pix for z, y, x]

    Returns:
        Tuple of slices: (z_slice, y_slice, x_slice) in pixels
    """
    # Convert start positions (um -> pix)
    start_idx = np.array([roi.z, roi.y, roi.x]) / spacing  # order: z, y, x
    start_idx = np.round(start_idx).astype(int)
    # Convert lengths (um -> pix)
    length_pix = np.array([roi.z_length, roi.y_length, roi.x_length]) / spacing
    length_pix = np.round(length_pix).astype(int)
    # Compute stop position (in pix)
    stop_idx = start_idx + length_pix

    # Build slices
    z_slice = slice(start_idx[0], stop_idx[0])
    y_slice = slice(start_idx[1], stop_idx[1])
    x_slice = slice(start_idx[2], stop_idx[2])

    return (z_slice, y_slice, x_slice)


def clip_region_stop_to_shape(
    region: Tuple[slice, slice, slice],
    shape: Tuple[int, int, int],
) -> Tuple[slice, slice, slice]:
    """
    Clip the stop values of a region (given as slices) so they do not exceed
    the bounds defined by the target shape.

    This is useful when slicing from a NumPy or Zarr array and the region
    may extend beyond the array's dimensions. Handles None values in the slices
    gracefully (interpreting them as 0 for start and the full dimension size for stop).

    Parameters
    ----------
    region : Tuple[slice, slice, slice]
        A tuple of slice objects representing the region to extract from a 3D array.
        Each slice can have `start` and `stop` values, which may be `None`.

    shape : Tuple[int, int, int]
        The shape of the array (z, y, x) to clip against. The clipped slices will
        ensure that stop <= corresponding shape dimension.

    Returns
    -------
    Tuple[slice, slice, slice]
        A new tuple of slices where the stop values are clipped to the array bounds,
        and None values for start/stop are replaced with default safe values.
    """
    region_clipped = []
    for sl, dim_size in zip(region, shape):
        start = sl.start if sl.start is not None else 0
        stop = min(sl.stop, dim_size) if sl.stop is not None else dim_size
        region_clipped.append(slice(start, stop))

    return tuple(region_clipped)


def get_region_shape(region: Tuple[slice, slice, slice]) -> Tuple[int, int, int]:
    """
    Compute the shape (z, y, x) of a region defined by slices.

    Parameters
    ----------
    region : Tuple[slice, slice, slice]
        Tuple of slice objects defining a region.

    Returns
    -------
    Tuple[int, int, int]
        The shape of the region, calculated as (stop - start) per axis.
    """
    shape = []
    for sl in region:
        if sl.start is None:
            raise ValueError("Cannot compute shape for region with no start value.")
        if sl.stop is None:
            raise ValueError("Cannot compute shape for region with no stop value.")
        start = sl.start
        stop = sl.stop
        shape.append(stop - start)

    return tuple(shape)


def save_new_label_image_with_overlap(
    new_nplabel: np.ndarray,
    zarr_url: str,
    new_label_name: str,
    region: Tuple[slice, slice, slice],
):
    """
    Write label numpy array to zarr in specific ROI region.
    new_npimg is a 3D numpy array.
    To be used within for loop over ROIs.
    Load on-disk region and combine with new array using element-wise max to handle potential overlapping regions
    that were already written to disk from previous ROIs.
    Apply to all channels
    """

    if new_nplabel.ndim != 3:
        raise ValueError(f"Expected 3D array, got {new_nplabel.ndim}D.")

    assert len(region) == 3  # region should have z,y,x coordinates

    # Load dask from disk, will contain rois of the previously processed objects within for loop
    image_url_level0 = os.path.join(zarr_url, "labels", new_label_name, "0")
    image_array = zarr.open_array(image_url_level0)

    if image_array.ndim != 3:
        raise ValueError(f"Expected 3D array, got {image_array.ndim}D.")

    # Load region of current object from disk, will include any previously processed neighboring objects
    try:
        region_on_disk = image_array[region]
    except IndexError:
        # This means region was padded and now extends beyond edges of full dask array
        # Clip region to max array size
        logger.info(
            f"Padded array extends beyond image boundary. Clipping stop coordinates of {region=} "
            f"to match array shape {image_array.shape[-3:]}"
        )
        region = clip_region_stop_to_shape(region, image_array.shape[-3:])
        region_on_disk = image_array[region]

        # Clip input numpy array to same shape
        region_shape = get_region_shape(region)
        z, y, x = region_shape
        new_nplabel = new_nplabel[:z, :y, :x]

    # Check that dimensions of rois match
    if region_on_disk.shape != new_nplabel.shape:
        raise ValueError(
            f"Shape of new label {new_nplabel.shape} must match region {region_on_disk.shape}. "
        )

    # Combine new image with the region that is already on disk using element-wise max
    result = np.fmax(new_nplabel, region_on_disk)

    # Store 0-th level of region on disk
    # TODO: Is on-disk chunking preserved here?
    da.array(result).to_zarr(
        url=image_array,
        region=region,
        overwrite=True,
        compute=True,
    )

    return


def load_channel_image(
    channel_input_model,
    dask_array,
    zarr_url,
    level,
    roi_table,
    roi_adata,
    xycoars_raw,
    pixmeta_raw,
):

    ##############
    # Load Channel images  ###
    ##############

    # Find channel index for channel
    tmp_channel: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=zarr_url,
        wavelength_id=channel_input_model.wavelength_id,
        label=channel_input_model.label,
    )

    channel_id = tmp_channel.index

    # Load channel data
    ch_dask_raw = dask_array[channel_id]

    ch_idlist_raw = convert_ROI_table_to_indices(
        roi_adata,
        level=level,
        coarsening_xy=xycoars_raw,
        full_res_pxl_sizes_zyx=pixmeta_raw,
    )

    check_valid_ROI_indices(ch_idlist_raw, roi_table)

    return ch_dask_raw, ch_idlist_raw


def load_label_rois(
    zarr_url,
    label_name,
    roi_table,
    level,
):

    # Lazily load zarr array for reference cycle
    # load well image as dask array e.g. for nuclear segmentation
    try:
        label_dask = da.from_zarr(f"{zarr_url}/labels/{label_name}/{level}")
    except ArrayNotFoundError as e:
        logger.warning(
            "Label not found, exit from the task for this zarr url.\n"
            f"Original error: {str(e)}"
        )
        return {}

    # Read ROIs of objects
    path = f"{zarr_url}/tables/{roi_table}"
    roi_adata = ad.read_zarr(path)
    roi_attrs = get_zattrs(path)

    # Read Zarr metadata
    label_ngffmeta = load_NgffImageMeta(f"{zarr_url}/labels/{label_name}")
    label_xycoars = (
        label_ngffmeta.coarsening_xy
    )  # need to know when building new pyramids
    label_pixmeta = label_ngffmeta.get_pixel_sizes_zyx(level=level)

    # Create list of indices for 3D ROIs spanning the entire Z direction
    # Note that this ROI list is generated based on the input ROI table; if the input ROI table is for the group_by
    # objects, then label regions will be loaded based on the group_by ROIs
    roi_idlist = convert_ROI_table_to_indices(
        roi_adata,
        level=level,
        coarsening_xy=label_xycoars,
        full_res_pxl_sizes_zyx=label_pixmeta,
    )

    instance_key = roi_attrs["instance_key"]  # e.g. "label"

    # NGIO FIX, TEMP
    # Check that ROI_table.obs has the right column and extract label_value
    if instance_key not in roi_adata.obs.columns:
        if roi_adata.obs.index.name == instance_key:
            # Workaround for new ngio table
            roi_adata.obs[instance_key] = roi_adata.obs.index
        else:
            raise ValueError(
                f"Table at {path=} has {instance_key=} "
                f" missing in {roi_adata.obs.columns=}"
            )

    check_valid_ROI_indices(roi_idlist, roi_table)

    if len(roi_idlist) == 0:
        logger.warning("Well contains no objects")

    return label_dask, roi_adata, roi_idlist, label_ngffmeta, label_pixmeta


def load_image_array(zarr_url, level):
    ngffmeta_raw = load_NgffImageMeta(f"{zarr_url}")
    xycoars_raw = ngffmeta_raw.coarsening_xy
    pixmeta_raw = ngffmeta_raw.get_pixel_sizes_zyx(level=level)

    img_array = da.from_zarr(f"{zarr_url}/{level}")

    return img_array, ngffmeta_raw, xycoars_raw, pixmeta_raw


def load_seg_and_raw_region(
    label_dask, channel_dask, label_idlist, channel_idlist, row_integer, compute
):
    # Load label image of label_name object as numpy array
    seg_numpy = load_region(
        data_zyx=label_dask,
        region=convert_indices_to_regions(label_idlist[row_integer]),
        compute=compute,
    )

    # Load raw image of specific channel for object
    raw_numpy = load_region(
        data_zyx=channel_dask,
        region=convert_indices_to_regions(channel_idlist[row_integer]),
        compute=compute,
    )

    return seg_numpy, raw_numpy
