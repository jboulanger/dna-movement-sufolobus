import argparse
from pathlib import Path
import dnasufo


def process_file(args):
    """Process file as a command line"""
    dnasufo.process_file(Path(args.root), Path(args.dst), args.index)


def list_files(args):
    """List files as a command line"""
    dnasufo.list_files(Path(args.root), Path(args.dst), [args.membrane, args.dna])


parser = argparse.ArgumentParser("dna-sufolobus")
subparsers = parser.add_subparsers()

subparser_list = subparsers.add_parser("list")
subparser_list.add_argument(
    "--root", type=Path, required=False, help="path to the source data"
)
subparser_list.add_argument(
    "--dst", type=Path, required=True, help="path to the destination result"
)
subparser_list.add_argument(
    "--membrane", type=int, default=0, help="index of membrane channel"
)
subparser_list.add_argument("--dna", type=int, default=1, help="index of dna channel")
subparser_list.set_defaults(func=list_files)

subparser_process = subparsers.add_parser("process")
subparser_process.add_argument(
    "--root", type=Path, required=False, help="root source data"
)
subparser_process.add_argument(
    "--dst", type=Path, required=True, help="path to the destination result"
)
subparser_process.add_argument(
    "--index", type=int, required=True, help="index of the filelist"
)
subparser_process.set_defaults(func=process_file)
args = parser.parse_args()
args.func(args)
