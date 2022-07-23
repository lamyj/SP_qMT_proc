# //////////////////////////////////////////////////////////////////////////////
# // L. SOUSTELLE, PhD, Aix Marseille Univ, CNRS, CRMBM, Marseille, France
# // 2021/02/06
# // Contact: lucas.soustelle@univ-amu.fr
# //////////////////////////////////////////////////////////////////////////////

import argparse
import collections
import logging
import sys
import textwrap
import time

import nibabel
import numpy

from . import utils
from . import _MTsat

def main(args):
    logging.basicConfig(
        level=args.verbosity.upper(),
        format="%(levelname)s - %(name)s: %(message)s")
    
    # Create and check equence parameters
    SEQparx_NT = collections.namedtuple("SEQparx_NT","TR1 TR FA")
    SEQparx = SEQparx_NT(
        TR1=args.SEQparx[0]*1e-3, TR=args.SEQparx[1]*1e-3,
        FA=numpy.radians(args.SEQparx[2]))
    
    for value in SEQparx:
        if value<0:
            parser.error("All SEQparx values should be positive")
    
    logging.info(textwrap.dedent("""\
        Summary of input sequence parameters:\n
        \tMT preparation module duration: {:.1f} ms
        \tSequence TR: {:.1f} ms
        \tReadout flip angle: {:.1f} deg
        """).format(SEQparx.TR1*1e3, SEQparx.TR*1e3, numpy.degrees(SEQparx.FA)))
    
    # Check input data
    if args.MT.ndim != 4:
        parser.error("Volume {} is not 4D".format(args.MT.get_filename()))
    if args.B1 is None:
        logging.warning("No B1 map provided (this is highly not recommended)")
    
    # Get MT data
    MT_data = args.MT.get_fdata()
    MT0_data = MT_data[..., 0]
    MTw_data = MT_data[..., 1]
    shape = MT0_data.shape
    
    # Get T1 and optional B1 data
    T1_map = args.T1.get_fdata()
    B1_map = args.B1.get_fdata() if args.B1 is not None else numpy.ones(shape)
    
    # Compute the mask of usable voxels
    mask = args.mask.get_fdata() if args.mask is not None else numpy.ones(shape)
    mask = (mask != 0) & ~numpy.isnan(T1_map) & (T1_map != 0) & (MT0_data != 0)
    
    # Estimation
    T1_data = T1_map[mask]
    cosFA_RO = numpy.cos(SEQparx.FA * B1_map[mask])
    E1 = numpy.exp(-SEQparx.TR1 / T1_data)
    E2 = numpy.exp(-(SEQparx.TR-SEQparx.TR1) / T1_data)
    Mz_MT0 = (1-E1*E2) / (1-E1*E2*cosFA_RO)
    signal_ratio = MTw_data[mask]/MT0_data[mask]
    data = numpy.stack((E1, E2, cosFA_RO, Mz_MT0, signal_ratio), axis=-1)
    
    start_time = time.time()
    fitted = _MTsat.fit(data, args.xtol)
    stop_time = time.time()
    logging.debug("Done in {} seconds".format(stop_time - start_time))
    
    MTsat_map = numpy.zeros(shape)
    MTsat_map[mask] = fitted*100
    # MTsat_map[(MTsat_map < 0) | (MTsat_map > 1000)] = 0
    
    # Create & save NIfTI(s)
    nibabel.save(nibabel.Nifti1Image(MTsat_map, args.MT.affine), args.MTsat)
    if args.MTsatB1sq is not None and args.B1 is not None:
        MTsatB1sq_map = numpy.zeros(shape)
        MTsatB1sq_map[mask] = MTsat_map[mask]/B1_map[mask]**2
        nibabel.save(
            nibabel.Nifti1Image(MTsatB1sq_map, args.MT.affine), args.MTsatB1sq)

def setup(subparsers):
    description = (
        "Compute MT saturation map [1,2] from an MT-prepared SPGR experiment. "
            "Outputs are in percentage unit.\n"
        "References:\n"
        "\t [1] G. Helms et al., High-resolution maps of magnetization "
            "transfer with inherent correction for RF inhomogeneity and T1 "
            "relaxation obtained from 3D FLASH MRI, MRM 2008;60:1396-1407\n"
        "\t [2] G. Helms et al., Modeling the influence of TR and excitation "
            "flip angle on the magnetization transfer ratio (MTR) in human "
            "brain obtained from 3D spoiled gradient echo MRI. "
            "MRM 2010;64:177-185")
    
    parser = subparsers.add_parser(
        "MTsat", aliases=["mtsat"], help="MT saturation map", 
        description=description, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        "MT", type=utils.image_stack_argument,
        help="Input (MT0/MTw) NIfTI path, 4D file or comma-separated 3D files")
    parser.add_argument(
        "T1", type=utils.image_argument, help="Input T1 (in sec) NIfTI path")
    parser.add_argument("MTsat", help="Output MTsat NIfTI path")
    parser.add_argument(
        "SEQparx", nargs="?", type=utils.tuple_argument(3*[float]),
        help="Sequence parameters (comma-separated), in this order:\n"
            "\t1) MT preparation module duration (ms)\n"
            "\t2) Sequence TR (ms)\n"
            "\t3) Readout flip angle (deg)\n"
            "\te.g. 10.0,43.0,10.0")
    parser.add_argument(
        "--MTsatB1sq", 
        help="Output MTsat image normalized by squared B1 NIfTI path")
    parser.add_argument(
        "--B1", type=utils.image_argument, 
        help="Input B1 map (in absolute unit) NIfTI path")
    parser.add_argument(
        "--mask", type=utils.image_argument,
        help="Input binary mask NIfTI path")
    parser.add_argument(
        "--xtol", type=float, default=1e-6,
        help="x tolerance for root finding (default: 1e-6)")
    utils.add_verbosity(parser)
    
    return parser
