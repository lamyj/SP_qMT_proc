import argparse
import os
import subprocess
import sys

import nibabel

def get_physCPU_number():
    """ Return the number of physical CPU cores.
    """
    
    # from joblib source code (commit d5c8274)
    # https://github.com/joblib/joblib/blob/d5c8274/joblib/externals/loky/backend/context.py#L220-L246
    if sys.platform == "linux":
        cpu_info = subprocess.run(
            "lscpu --parse=core".split(" "), capture_output=True)
        cpu_info = cpu_info.stdout.decode("utf-8").splitlines()
        cpu_info = {line for line in cpu_info if not line.startswith("#")}
        cpu_count_physical = len(cpu_info)
    elif sys.platform == "win32":
        cpu_info = subprocess.run(
            "wmic CPU Get NumberOfCores /Format:csv".split(" "),
            capture_output=True)
        cpu_info = cpu_info.stdout.decode('utf-8').splitlines()
        cpu_info = [l.split(",")[1] for l in cpu_info
                    if (l and l != "Node,NumberOfCores")]
        cpu_count_physical = sum(map(int, cpu_info))
    elif sys.platform == "darwin":
        cpu_info = subprocess.run(
            "sysctl -n hw.physicalcpu".split(" "), capture_output=True)
        cpu_info = cpu_info.stdout.decode('utf-8')
        cpu_count_physical = int(cpu_info)
    else:
        raise NotImplementedError(
            "unsupported platform: {}".format(sys.platform))
    if cpu_count_physical < 1:
            raise ValueError(
                "found {} physical cores < 1".format(cpu_count_physical))
    return cpu_count_physical

def tuple_argument(types):
    def parser(value):
        items = value.split(",")
        if len(items) != len(types):
            raise argparse.ArgumentTypeError(
                "Wrong arguments count: expected {}, got {}".format(
                    len(types), len(items)))
        items = [type_(item) for type_, item in zip(types, items)]
        return items
    return parser

def image_argument(value):
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError("No such file: {}".format(value))
    try:
        image = nibabel.load(value)
    except Exception as e:
        raise argparse.ArgumentTypeError(str(e))
    else:
        return image

def add_verbosity(parser):
    """ Add verbosity argument to an argument parser
    """
    
    parser.add_argument(
        "--verbosity", "-v",
        choices=["warning", "info", "debug"], default="warning")
