import argparse
import multiprocessing

import nibabel
import numpy

from .. import utils, vfa

def main(args):  
    TR = args.TR*1e-3
    FA = numpy.radians(args.FA)
    
    if args.FitType == "default":
        args.FitType = "NLS" if args.VFA.shape[-1] > 2 else "LLS"
    
    shape = args.VFA.shape[:-1]
    
    B1_map = args.B1.get_fdata() if args.B1 is not None else numpy.ones(shape)
    
    mask = args.mask.get_fdata() if args.mask is not None else numpy.ones(shape)
    mask = (mask != 0)
    
    VFA_data = args.VFA.get_fdata()[mask]
    B1_data = B1_map[mask]
    FA_data = B1_data[:, None]*FA
    
    fit_functions = {
        "LLS": (vfa.linear_fit, linear_fit),
        "NLS": (vfa.non_linear_fit, non_linear_fit)
    }
    scalar_fit, parallel_fit = fit_functions[args.FitType]
    if args.nworkers > 1:
        with multiprocessing.Pool(args.nworkers) as pool:
            data = numpy.empty((len(FA_data), 5))
            data[:, 0:2] = FA_data
            data[:, 2:4] = VFA_data
            data[:, 4] = TR
            fitted = pool.map(
                parallel_fit, numpy.array_split(data, 10*args.nworkers))
        fitted_T1 = numpy.concatenate([x[0] for x in fitted])
        fitted_S0 = numpy.concatenate([x[1] for x in fitted])
    else:
        fitted_T1, fitted_S0 = scalar_fit(FA_data, VFA_data, TR)
    
    outliers = (fitted_T1 < 0) | (fitted_T1 > 10) | (fitted_S0 < 0)
    fitted_T1[outliers] = 0
    fitted_S0[outliers] = 0
    
    T1_map = numpy.zeros(shape)
    T1_map[mask] = fitted_T1
    nibabel.save(nibabel.Nifti1Image(T1_map, args.VFA.affine), args.T1)
    
    if args.S0:
        S0_map = numpy.zeros(shape)
        S0_map[mask] = fitted_S0
        nibabel.save(nibabel.Nifti1Image(S0_map, args.VFA.affine), args.S0)

def linear_fit(data):
    """ Entry point for multi-processing linear fit. This is required, since
        VFA.linear_fit is not picklable.
    """
    
    FA_data = data[:, 0:2]
    VFA_data = data[:, 2:4]
    TR = data[0, 4]
    return vfa.linear_fit(FA_data, VFA_data, TR)

def non_linear_fit(data):
    """ Entry point for multi-processing non-linear fit. This is required, since
        VFA.non_linear_fit is not picklable.
    """ 
    
    FA_data = data[:, 0:2]
    VFA_data = data[:, 2:4]
    TR = data[0, 4]
    return vfa.non_linear_fit(FA_data, VFA_data, TR)

def setup(subparsers):
    description = (
        "Fit T1 from a Variable Flip Angle (VFA) protocol.\n"
        "Returns maps of T1 (s) and S0 from: S=S0*(1-E1)*sin(FA)/(1-E1*cos(FA))\n"
        "References:\n"
        "\t [1] Chang et al., Linear least-squares method for unbiased"
            "estimation of T1 from SPGR signals, MRM 2008;60:496-501")
    
    parser = subparsers.add_parser(
        "VFA", aliases=["vfa"], help="T1 map", 
        description=description, formatter_class=argparse.RawTextHelpFormatter)
                         
    parser.add_argument(
        "VFA", type=utils.image_stack_argument,
        help="Input VFA NIfTI path, 4D file or comma-separated 3D files")
    parser.add_argument("T1", help="Output T1 NIfTI path.")
    parser.add_argument(
        "--TR", required=True, type=float,
        help="Sequence Time to Repetition (ms).")
    parser.add_argument(
        "--FA", type=utils.tuple_argument(float), required=True,
        help="Comma-separated Flip Angles (degrees)")
    parser.add_argument(
        "--B1", type=utils.image_argument,
        help="Input B1 map NIfTI path. Highly recommended")
    parser.add_argument(
        "--mask", type=utils.image_argument,
        help="Input Mask binary NIfTI path.")
    parser.add_argument("--S0", help="Output S0 NIfTI path.")
    parser.add_argument(
        "--FitType", choices=["default", "NLS", "LLS"], default="default",
        help="Fitting type\n"
            "\tNLS: Nonlinear Least-Square (default for > 2 points VFA)\n"
            "\tLLS: Linear Least-Square (default to 2 points VFA)")
    parser.add_argument("--nworkers", nargs="?", type=int, default=1)
    utils.add_verbosity(parser)
    
    return parser
