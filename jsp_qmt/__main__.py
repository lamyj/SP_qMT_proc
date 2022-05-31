import argparse
import logging
import sys

from . import fit_MTsat_CLI

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Available commands")
    command_parsers = {}
    for name in ["MTsat"]:
        module = globals()["fit_{}_CLI".format(name)]
        subparser = module.setup(subparsers)
        subparser.set_defaults(action=module.main)
        command_parsers[module.main] = subparser
    arguments = parser.parse_args()
    
    if "action" not in arguments:
        parser.print_help()
        return 1
    
    try:
        arguments.action(arguments)
    except Exception as e:
        if getattr(logging, arguments.verbosity.upper()) <= logging.DEBUG:
            raise
        else:
            command_parsers[arguments.action].error(e)

if __name__ == "__main__":
    sys.exit(main())
