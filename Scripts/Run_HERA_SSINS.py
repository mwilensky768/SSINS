from SSINS import SS, INS, version
from SSINS import Catalog_Plot as cp
import numpy as np
import argparse

parser = parser.ArgumentParser()
parser.add_argument("-f", "--filename", help="The visibility file to process")
parser.add_argument("-s", "--streak_sig", help="The desired streak significance threshold")
parser.add_argument("-o", "--other_sig", help="The desired significance threshold for other shapes")
parser.add_argument("-p", "--prefix", help="The prefix for output files")
parser.add_argument("-N", "--N_samp_thresh", help="The N_samp_thresh parameter for the match filter")
args = parser.parse_args()

version_info_list = ['%s: %s, ' % (key, version.version_info[key]) for key in version.version_info]
version_hist_substr = reduce(lambda x, y: x + y, version_info_list)

# Make the SS object
ss = SS()
ss.read(args.filename, ant_str='cross')

# Make the INS object
ins = INS(ss)

# Write the raw data and z-scores to h5 format
ins.write(args.prefix)
ins.write(args.prefix, output_type='z_score')

# Flag FM radio
where_FM = np.where(np.logical_and(ins.freq_array > 87.5e6, ins.freq_array < 108e6))
ins.metric_array[:, where_FM] = np.ma.masked
ins.metric_ms = ins.mean_subtract()
ins.history += "Manually flagged the FM band. "

# Make a filter with specified settings
dab_width = 1.536e6
dab_freqs = np.arange(214e6, 230e6, dab_width)
dab_dict = {'DAB%i' % ind: [dab_freqs[ind], dab_freqs[ind + 1]] for ind in range(len(dab_freqs) - 1)}

shape_dict = {'TV4': [174e6, 182e6],
              'TV5': [182e6, 190e6],
              'TV6': [190e6, 198e6],
              'TV7': [198e6, 206e6],
              'TV8': [206e6, 214e6],
              'TV9': [214e6, 222e6],
              'TV10': [222e6, 230e6],
              'TV11': [230e6, 238e6],
              'LB1': [45e6, 57e6],
              'LB2': [63e6, 70e6],
              'LB3': [73e6, 80e6],
              'MB1': [128e6, 133e6],
              'MB2': [140e6, 147e6],
              'MB3': [153e6, 160e6],
              'MB4': [165e6, 170e6]}
for shape in dab_dict:
    shape_dict[shape] = dab_dict[shape]
sig_thresh = {shape: args.other_sig for shape in shape_dict}
sig_thresh['narrow'] = args.other_sig
sig_thresh['streak'] = args.streak_sig
mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict, N_samp_thresh=args.N_samp_thresh)

# Do flagging
mf.apply_match_test(ins)
ins.history += "Flagged using apply_match_test on SSINS %s." % version_hist_substr

# Write outputs
ins.write(args.prefix, output_type='mask')
ins.write(args.prefix, output_type='flags')
ins.write(args.prefix, output_type='match_events')
