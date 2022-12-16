""" Script to grab LLC model data """

from IPython import embed

'''
LLC docs: https://mitgcm.readthedocs.io/en/latest/index.html
'''

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Analyze GP data')
    parser.add_argument("inp_file", type=str, help="Inptut file [JSON]")
    #parser.add_argument("--model", type=str, default='LLC4320',
    #                    help="LLC Model name.  Allowed options are [LLC4320]")
    #parser.add_argument("--var", type=str, default='Theta',
    #                    help="LLC data variable name.  Allowed options are [Theta, U, V, Salt]")
    #parser.add_argument("--istart", type=int, default=0,
    #                    help="Index of model to start with")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import numpy as np
    import os

    from glider_gp import utils as glider_utils
    from glider_gp import calypso

    # Load input file
    pargs = glider_utils.loadjson(pargs.inp_file)

    # Load the data
    if pargs['dataset'] == 'calypso':
        data = calypso.load_for_gp(pargs)

    embed(header='44 of analyze.py')

    # Prep for GP