from __future__ import absolute_import, division, print_function

import argparse
from SSINS import Catalog_Plot as cp
from SSINS import SS
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('obs', action='store', help='How the observation will be referred to')
parser.add_argument('inpath', action='store', help='The path to the data file, and the file_type')
parser.add_argument('outpath', action='store', help='The base directory for saving all outputs')
args = parser.parse_args()

# Here is a dictionary for the RFI class keywords

data_kwargs = {'read_kwargs': {'file_type': 'uvfits', 'ant_str': 'cross'},
               'obs': args.obs,
               'inpath': args.inpath,
               'outpath': args.outpath,
               'bad_time_indices': [0, -1, -2, -3]}

# The type of catalog you would like made - options are 'INS', 'VDH', 'MF', and 'ES'
catalog_types = ['INS', 'MF', 'ES']


catalog_data_kwargs = {'INS': {},
                       'VDH': {},
                       'MF': {'sig_thresh': 4.5,
                              'shape_dict': {'TV%i' % (k + 6): [(1.74 + 0.07 * k) * 10 ** 8,
                                                                (1.81 + 0.07 * k) * 10 ** 8] for k in range(3)}},
                       'ES': {}}

catalog_plot_kwargs = {'INS': {},
                       'VDH': {},
                       'MF': {},
                       'ES': {}}

"""
Do not edit things beneath this line!
"""

sky_sub = SS(**data_kwargs)

for cat in catalog_types:
    getattr(sky_sub, '%s_prepare' % (cat))(**catalog_data_kwargs[cat])
    getattr(cp, '%s_plot' % (cat))(getattr(sky_sub, cat), **catalog_plot_kwargs[cat])
sky_sub.save_data()
sky_sub.save_meta()
