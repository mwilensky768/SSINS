import argparse
from rfipy import RFI
import Catalog_Plot as cp

parser = argparse.ArgumentParser()
parser.add_argument('obs', action='store', help='How the observation will be referred to')
parser.add_argument('inpath', action='store', help='The path to the data file')
parser.add_argument('outpath', action='store', help='The base directory for saving all outputs')
parser.add_argument('filetype', action='store', help='The type of file to be read in by pyuvdata')
data_args = parser.parse_args()

# Here is a dictionary for the RFI class keywords
data_kwargs = {}

# The type of catalog you would like made - options are 'INS', 'VDH', and 'BA'
catalog_types = ['INS', 'VDH']

catalog_data_kwargs = {}
catalog_plot_kwargs = {}

"""
Do not edit things beneath this line!
"""

rfi = RFI(args.obs, args.inpath, args.outpath, args.filetype, **data_kwargs)

for cat in catalog_types:
    catalog_data = getattr(rfi, '%s_prepare' % (cat))(**catalog_data_kwargs[catalog_types])
    getattr(cp, '%s_plot' % (cat))(catalog_data, **catalog_plot_kwargs[catalog_types])
