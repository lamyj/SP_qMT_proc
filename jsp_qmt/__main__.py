import argparse
import logging
import sys

from . import cli

def main(args=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Available commands")
    command_parsers = {}
    for name in ["MTsat", "SPqMT", "VFA"]:
        module = getattr(cli, name.lower())
        subparser = module.setup(subparsers)
        subparser.set_defaults(action=module.main)
        command_parsers[module.main] = subparser
    arguments = parser.parse_args(args)
    
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
