from __future__ import absolute_import, division, print_function

import argparse
from SSINS import Catalog_Plot as cp
from SSINS import SS
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('obs', action='store', help='How the observation will be referred to')
parser.add_argument('inpath', action='store', help='The path to the data file, and the file_type')
parser.add_argument('outpath', action='store', help='The base directory for saving all outputs')
parser.add_argument('-t', action='store', nargs='*', help='The desired catalogs.', required=True)
parser.add_argument('--ft', action='store', help='The file type')
parser.add_argument('--fc', action='store', help='Path to numpy loadable boolean array of frequencies to read in')
args = parser.parse_args()

# Here is a dictionary for the RFI class keywords

data_kwargs = {'read_kwargs': {'file_type': args.ft, 'ant_str': 'cross'},
               'obs': args.obs,
               'inpath': args.inpath,
               'outpath': args.outpath}

if args.fc is not None:
    data_kwargs['read_kwargs']['freq_chans'] = np.load(args.fc)

catalog_data_kwargs = {'INS': {},
                       'VDH': {},
                       'MF': {'shape_dict': {'TV6': [1.74e8, 1.81e8],
                                             'TV7': [1.81e8, 1.88e8],
                                             'TV8': [1.88e8, 1.95e8]}},
                       'ES': {}}

catalog_plot_kwargs = {'INS': {'ms_vmin': -5, 'ms_vmax': 5},
                       'VDH': {},
                       'MF': {},
                       'ES': {}}

"""
Do not edit things beneath this line!
"""

sky_sub = SS(**data_kwargs)
sky_sub.apply_flags(choice='original')


for cat in args.t:
    getattr(sky_sub, '%s_prepare' % (cat))(**catalog_data_kwargs[cat])
    getattr(cp, '%s_plot' % (cat))(getattr(sky_sub, cat), **catalog_plot_kwargs[cat])
sky_sub.save_data()
sky_sub.save_meta()
