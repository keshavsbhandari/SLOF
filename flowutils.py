"""
optflow.py
Optical flow I/O and visualization functions.
Written by Phil Ferriere
Licensed under the MIT License (see LICENSE for details)
Refs:
    - Per MPI-Sintel/flow_code/C/flowIO.h and flowIO.cpp:
    // the "official" threshold - if the absolute value of either
    // flow component is greater, it's considered unknown
    #define UNKNOWN_FLOW_THRESH 1e9
    // value to use to represent unknown flow
    #define UNKNOWN_FLOW 1e10
    // first four bytes, should be the same in little endian
    #define TAG_FLOAT 202021.25  // check for this when READING the file
    #define TAG_STRING "PIEH"    // use this when WRITING the file
    // ".flo" file format used for optical flow evaluation
    //
    // Stores 2-band float image for horizontal (u) and vertical (v) flow components.
    // Floats are stored in little-endian order.
    // A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
    //
    //  bytes  contents
    //
    //  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
    //          (just a sanity check that floats are represented correctly)
    //  4-7     width as an integer
    //  8-11    height as an integer
    //  12-end  data (width*height*2*4 bytes total)
    //          the float values for u and v, interleaved, in row order, i.e.,
    //          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
    - Numpy docs:
    ndarray.tofile(fid, sep="", format="%s")
    https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile
    numpy.fromfile(file, dtype=float, count=-1, sep='')
    https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.fromfile.html
"""

from __future__ import absolute_import, division, print_function
import os
import warnings
import numpy as np
import cv2
from skimage.io import imsave

def clean_dst_file(dst_file):
    """Create the output folder, if necessary; empty the output folder of previous predictions, if any
    Args:
        dst_file: Destination path
    """
    # Create the output folder, if necessary
    dst_file_dir = os.path.dirname(dst_file)
    if not os.path.exists(dst_file_dir):
        os.makedirs(dst_file_dir)

    # Empty the output folder of previous predictions, if any
    if os.path.exists(dst_file):
        os.remove(dst_file)


##
# I/O utils
##

TAG_FLOAT = 202021.25


def flow_read(src_file):
    """Read optical flow stored in a .flo, .pfm, or .png file
    Args:
        src_file: Path to flow file
    Returns:
        flow: optical flow in [h, w, 2] format
    Refs:
        - Interpret bytes as packed binary data
        Per https://docs.python.org/3/library/struct.html#format-characters:
        format: f -> C Type: float, Python type: float, Standard size: 4
        format: d -> C Type: double, Python type: float, Standard size: 8
    Based on:
        - To read optical flow data from 16-bit PNG file:
        https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py
        Written by Clément Pinard, Copyright (c) 2017 Clément Pinard
        MIT License
        - To read optical flow data from PFM file:
        https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/pfm.py
        Written by Ruoteng Li, Copyright (c) 2017 Ruoteng Li
        License Unknown
        - To read optical flow data from FLO file:
        https://github.com/daigo0927/PWC-Net_tf/blob/master/flow_utils.py
        Written by Daigo Hirooka, Copyright (c) 2018 Daigo Hirooka
        MIT License
    """
    # Read in the entire file, if it exists
    assert(os.path.exists(src_file))

    if src_file.lower().endswith('.flo'):

        with open(src_file, 'rb') as f:

            # Parse .flo file header
            tag = float(np.fromfile(f, np.float32, count=1)[0])
            assert(tag == TAG_FLOAT)
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]

            # Read in flow data and reshape it
            flow = np.fromfile(f, np.float32, count=h * w * 2)
            flow.resize((h, w, 2))

    elif src_file.lower().endswith('.png'):

        # Read in .png file
        flow_raw = cv2.imread(src_file, -1)

        # Convert from [H,W,1] 16bit to [H,W,2] float formet
        flow = flow_raw[:, :, 2:0:-1].astype(np.float32)
        flow = flow - 32768
        flow = flow / 64

        # Clip flow values
        flow[np.abs(flow) < 1e-10] = 1e-10

        # Remove invalid flow values
        invalid = (flow_raw[:, :, 0] == 0)
        flow[invalid, :] = 0

    elif src_file.lower().endswith('.pfm'):

        with open(src_file, 'rb') as f:

            # Parse .pfm file header
            tag = f.readline().rstrip().decode("utf-8")
            assert(tag == 'PF')
            dims = f.readline().rstrip().decode("utf-8")
            w, h = map(int, dims.split(' '))
            scale = float(f.readline().rstrip().decode("utf-8"))

            # Read in flow data and reshape it
            flow = np.fromfile(f, '<f') if scale < 0 else np.fromfile(f, '>f')
            flow = np.reshape(flow, (h, w, 3))[:, :, 0:2]
            flow = np.flipud(flow)
    else:
        raise IOError

    return flow


def flow_write(flow, dst_file):
    """Write optical flow to a .flo file
    Args:
        flow: optical flow
        dst_file: Path where to write optical flow
    """
    # Create the output folder, if necessary
    # Empty the output folder of previous predictions, if any
    clean_dst_file(dst_file)

    # Save optical flow to disk
    with open(dst_file, 'wb') as f:
        np.array(TAG_FLOAT, dtype=np.float32).tofile(f)
        height, width = flow.shape[:2]
        np.array(width, dtype=np.uint32).tofile(f)
        np.array(height, dtype=np.uint32).tofile(f)
        flow.astype(np.float32).tofile(f)
