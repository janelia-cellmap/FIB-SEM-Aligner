#!/usr/bin/env python
# -*- coding: utf-8 -*-
# binboxes.py
"""
CLI for binning the output of 07_materialize_boxes.sh

Copyright (c) 2017, David Hoffman
"""

import click
import os
import glob
import numpy as np
from skimage.io import imread
from skimage.external import tifffile as tif
import dask
import dask.multiprocessing
from dask.diagnostics import ProgressBar

# register dask progress bar
ProgressBar().register()


# @dask.delayed(pure=True)
# def lazy_imread(path_or_url):
#     """Lazily read in image data"""
#     return imread(path_or_url)


def parse_tile_num(path):
    """get the tile number from path"""
    split_path = path.split(os.path.sep)
    # get the tile number from path name
    return int(split_path[-3])


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False,
                                       readable=True, allow_dash=False))
@click.argument("dst", type=click.Path(exists=False, file_okay=False,
                                       writable=True, allow_dash=False))
@click.option("--binning", default=4, help="The number of z-slices to bin")
def cli(src, dst, binning):
    """
    Read in fibsem data and bin along z, cropping and
    binning along xy to be added later

    Read in all files of glob pattern src + "/**/0." bin by `binning`
    and save in dst
    """
    # get a sorted list of all the files
    click.echo("Searching for files in {} ... ".format(os.path.abspath(src)), nl=False)
    file_list = sorted(glob.glob(src + "/**/0/0.*", recursive=True),
                       key=parse_tile_num)
    click.echo("found {} files".format(len(file_list)))
    # make it an array
    file_array = np.asarray(file_list)
    num_digits = int(np.ceil(np.log10(file_array.size)))

    click.echo("Making DST directory")
    os.mkdir(dst)

    basename = os.path.abspath(dst + os.path.sep + "{{:0{}d}}.tif".format(num_digits))
    click.echo("basename is {}".format(basename))

    @dask.delayed(pure=True)
    def binner(files, i):
        """Bin the files and save them."""
        data = np.array([imread(file) for file in files])
        tif.imsave(basename.format(i), data.mean(0).astype(data.dtype),
                   compress=6)

    my_slice = slice(None, file_array.size - file_array.size % binning)
    to_compute = dask.delayed([
        binner(files, i)
        for i, files in enumerate(file_array[my_slice].reshape(-1, binning))
    ])
    # do the computation.
    to_compute.compute(get=dask.multiprocessing.get)


if __name__ == "__main__":
    cli()
