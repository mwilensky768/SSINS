import argparse
from rfipy import RFI
import Catalog_Plot as cp

parser = argparse.ArgumentParser()
parser.add_argument('obs', action='store', help='How the observation will be referred to')
parser.add_argument('outpath', action='store', help='The base directory for saving all outputs')
parser.add_argument('-f', 'inpath', nargs=2, action='store', help='The path to the data file, and the file_type')
data_args = parser.parse_args()

# Here is a dictionary for the RFI class keywords

data_kwargs = {}

# The type of catalog you would like made - options are 'INS', 'VDH', 'MF', and 'ES'
catalog_types = ['INS', ]

catalog_data_kwargs = {'INS': {},
                       'VDH': {},
                       'MF': {},
                       'ES': {}}

catalog_plot_kwargs = {'INS': {},
                       'VDH': {},
                       'MF': {},
                       'ES': {}}

"""
Do not edit things beneath this line!
"""

rfi = RFI(args.obs, args.outpath, args.inpath[0], args.filetype[1], **data_kwargs)

for cat in catalog_types:
    catalog_data = getattr(rfi, '%s_prepare' % (cat))(**catalog_data_kwargs[cat])
    getattr(cp, '%s_plot' % (cat))(catalog_data, **catalog_plot_kwargs[cat])
