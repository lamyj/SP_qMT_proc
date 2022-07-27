import argparse
import logging
import multiprocessing
import sys
import textwrap

import numpy
import nibabel
import scipy.integrate

from .. import get_pulse_average_and_rms, super_lorentzian, spqmt, utils

gamma = 267.513 * 1e6 # rad/s/T

def main(args):
    logging.basicConfig(
        level=args.verbosity.upper(),
        format="%(levelname)s - %(name)s: %(message)s")
    
    FAsat, delta_f, FWHM, HannApo, FAro = args.SATROparx
    FAsat, FAro = [numpy.radians(x) for x in [FAsat, FAro]]
    
    Tm, Ts, Tp, TR = [1e-3*x for x in args.SEQtimings]
    
    if args.qMTconstraintParx is None:
        presets = {
            # Yarnykh MRM 2012, 10.1002/mrm.23224
            1: ("3T adult human brain", 0.022, 10.0e-6, 19.0),
            # Soustelle et al. MRM 2021, 10.1002/mrm.28397
            2: ("7T adult mouse brain", 0.0129, 9.1e-6, 26.5)
        }
        RecoType, R1fT2f, T2r, R = presets[args.RecoTypePreset]
    else:
        RecoType = "User-defined"
        R1fT2f, T2r, R = args.qMTconstraintParx
            
    logging.info(textwrap.dedent("""\
        Summary of input sequence parameters:
        \t Saturation Flip Angle: {:.1f} deg
        \t Saturation pulse off-resonance frenquency: {:.1f} Hz
        \t Gaussian pulse FWHM: {:.1f} Hz
        \t Hann apodization: {}
        \t Readout Flip Angle: {:.1f} deg
        \t Saturation pulse duration: {:.2f} ms
        \t Interdelay Saturation pulse <--> Readout pulse: {:.2f} ms
        \t Readout pulse duration: {:.2f} ms
        \t Sequence Time to Repetition: {:.1f} ms
        
        Summary of constraint qMT parameters ({}):
        \t R1fT2f: {:.4f}
        \t T2r:\t {:.1f} us
        \t R:\t {:.1f} s-1""").format(
            numpy.degrees(FAsat), delta_f, FWHM, HannApo, numpy.degrees(FAro),
            Tm*1e3, Ts*1e3, Tp*1e3, TR*1e3,
            RecoType, R1fT2f, T2r*1e6, R))
    
    # Get MT data
    MT_data = args.MT.get_fdata()
    MT0_data = MT_data[..., 0]
    MTw_data = MT_data[..., 1]
    shape = MT0_data.shape
    
    # Get T1 and optional B0 and B1 data
    T1_map = args.T1.get_fdata()
    B0_map = args.B0.get_fdata() if args.B0 is not None else numpy.zeros(shape)
    B1_map = args.B1.get_fdata() if args.B1 is not None else numpy.ones(shape)
    
    # Compute the mask of usable voxels
    mask = args.mask.get_fdata() if args.mask is not None else numpy.ones(shape)
    mask = (mask != 0) & (MT0_data != 0) & numpy.isfinite(T1_map) & (T1_map != 0)
    
    data = prepare_fit_data(
        FAsat, delta_f, FWHM, HannApo, FAro, Tm, Ts, Tp, TR,
        R1fT2f, T2r, R,
        T1_map[mask], B0_map[mask], B1_map[mask],
        MT0_data[mask], MTw_data[mask])

    if args.nworkers > 1:
        with multiprocessing.Pool(args.nworkers) as pool:
            fitted = pool.map(fit, numpy.array_split(data, 10*args.nworkers))
        fitted = numpy.concatenate(fitted)
    else:
        fitted = spqmt.fit(data)
    
    MPF_map = numpy.zeros(shape)
    MPF_map[mask] = fitted
    MPF_map = MPF_map / (1 + MPF_map)
    MPF_image = nibabel.Nifti1Image(MPF_map, args.MT.affine)
    nibabel.save(MPF_image, args.MPF)

def prepare_fit_data(
        FAsat, delta_f, FWHM, HannApo, FAro, Tm, Ts, Tp, TR,
        R1fT2f, T2r, R,
        T1, B0, B1, MT0, MTw):
    
    # Compute w1RMS nominal from the shaped pulse
    average, rms = get_pulse_average_and_rms(Tm, FWHM, HannApo)
    B1peak_nom = FAsat / (gamma*average*Tm)
    w1RMS_nom = gamma*B1peak_nom * rms

    # G(delta_f)
    if any(B0 != 0.0): # different G for voxels
        delta_f_corr = delta_f + B0
        G = super_lorentzian(T2r, delta_f_corr)
    else: # same G for all voxels
        G = super_lorentzian(T2r, delta_f)
    
    Wb = (numpy.pi * w1RMS_nom**2 * G) * B1**2

    R1 = 1/T1
    Wf = (w1RMS_nom/(2*numpy.pi))**2 / R1fT2f / (delta_f+B0)**2 * R1 * B1**2
    
    FAro_corr = B1*FAro
    Tr = TR - Ts - Tm - Tp
    y = MTw / MT0
    
    data = numpy.empty((Wb.shape[0], 9))
    for index, array in enumerate([Wb, Wf, FAro_corr, R1, R, Ts, Tm, Tr, y]):
        data[:, index] = array
    
    return data

def fit(data):
    """ Entry point for multi-processing fit. This is required, since _SPqMT.fit
        is not picklable.
    """ 
    return spqmt.fit(data)

def setup(subparsers):
    description = (
        "Fit a Macromolecular Proton Fraction (MPF) map from a Single-Point "
            "qMT protocol.\n"
        "Notes:\n"
        "\t1) The implemented saturation pulse is gaussian shaped with a "
            "user-defined FWHM.\n"
        "\tA Hann apodization is made possible, and a FWHM of 0.0 Hz yields a "
            "pure Hann-shaped pulse.\n"
        "\t  - Siemens' users: parameters are straightforwardly the same as "
            "in the Special Card interface from the greMT C2P sequence.\n"
        "\t  - Bruker's users: \"gauss\" pulse is a pure Gauss pulse with an "
            "FWHM of 218 Hz (differ from ParaVision's UI value).\n"
        "\t2) As in Yarnykh's original paper about SP-qMT [1], an assumption "
            "is made such that R1f = R1b = R1.\n"
        "\t3) The MT0 image to be provided can be computed from a 2-points VFA "
            "protocol [2], and synthetized via the synt_MT0_SPqMT.py script.\n"
        "\t4) B0 correction is not essential for SP-MPF mapping, but B1 is [3].\n"
        "References:\n"
        "\t[1] V. Yarnykh, Fast macromolecular proton fraction mapping from a "
            "single off-resonance magnetization transfer measurement, "
            "MRM 2012;68:166-178\n"
        "\t[2] V. Yarnykh, Time-efficient, high-resolution, whole brain "
            "three-dimensional macromolecular proton fraction mapping, "
            "MRM 2016;75:2100-2106\n"
        "\t[3] V. Yarnykh et al., Scan–Rescan Repeatability and Impact of B0 "
            "and B1 Field Nonuniformity Corrections in Single‐Point "
            "Whole‐Brain Macromolecular Proton Fraction Mapping, "
            "JMRI 2020;51:1789-1798\n"
        "\t[4] L. Soustelle et al., Determination of optimal parameters for "
            "3D single‐point macromolecular proton fraction mapping at 7T "
            "in healthy and demyelinated mouse brain, MRM 2021;85:369-379\n"
    )
    
    parser = subparsers.add_parser(
        "SPqMT", aliases=["spqmt"], help="Macromolecular Proton Fraction", 
        description=description, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        "MT", type=utils.image_stack_argument,
        help="Input (MT0/MTw) NIfTI path, 4D file or comma-separated 3D files")
    parser.add_argument(
        "T1", type=utils.image_argument, help="Input T1 NIfTI path")
    parser.add_argument("MPF", help="Output MPF NIfTI path")
    parser.add_argument(
        "SEQtimings", type=utils.tuple_argument(4*[float]),
        help=textwrap.dedent("""\
            Sequence timings in ms (comma-separated), in this order:
            \t1) Saturation pulse duration
            \t2) Interdelay between Saturation pulse and Readout pulse
            \t3) Readout pulse duration
            \t4) Sequence Time to Repetition (TR)
            e.g. 12.0,2.0,0.2,30.0"""))
    parser.add_argument(
        "SATROparx", type=utils.tuple_argument(3*[float]+[bool, float]),
        help=textwrap.dedent("""\
            Saturation and Readout pulse parameters (comma-separated), in this order:
            \t1) Saturation pulse Flip Angle (deg)
            \t2) Saturation pulse off-resonance frequency (Hz)
            \t3) Gaussian saturation pulse FWHM (Hz)
            \t4) Hann apodization (boolean; default: 1)
            \t5) Readout Flip Angle (deg)
            e.g. 560.0,4000.0,200.0,1,10.0"""))
    parser.add_argument(
        "--B1", type=utils.image_argument,
        help="Input B1 map NIfTI path. Highly recommended")
    parser.add_argument(
        "--B0", type=utils.image_argument, 
        help="Input B0 map NIfTI path")
    parser.add_argument(
        "--mask", type=utils.image_argument,
        help="Input Mask binary NIfTI path")
    parser.add_argument(
        "--RecoTypePreset", type=int, choices=[1,2], default=1,
        help=textwrap.dedent("""\
            SP-qMT reconstruction type (integer):
            \t1: Adult human brain 3T [1,2] (default)
            2: Adult mouse brain 7T [4]"""))
    parser.add_argument(
        "--qMTconstraintParx", type=utils.tuple_argument(3*[float]),
        help="Constained parameters for SP-qMT estimation (comma-separated; "
                "overrides --RecoTypePreset) in this order:\n"  
            "\t1) R1fT2f\n"
            "\t2) T2r (s)\n"
            "\t3) R (s^-1)\n"
            "e.g. 0.022,10.0e-6,19")
    parser.add_argument("--nworkers", type=int, default=1)
    utils.add_verbosity(parser)
    
    return parser
