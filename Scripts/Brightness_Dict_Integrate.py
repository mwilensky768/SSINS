from __future__ import division, absolute_import, print_function

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--bright_dict', help='Path to a pickled brightness dictionary.', required=True)
parser.add_argument('-o', '--outdir', help='The directory to put the outputs in.', required=True)
args = parser.parse_args()

with open(args.bright_dict, 'rb') as file:
    bright_dict = pickle.load(file)

sig_thresh = bright_dict.keys()[0]
subdict = bright_dict[sig_thresh]
shapes = subdict.keys()
obslist = bright_dict[sig_thresh][shapes[0]].keys()

bright_dict_total = {obs: np.sum([bright_dict[sig_thresh][shape][obs] for shape in shapes]) for obs in obslist}

with open('%s/bright_dict_total.pik' % args.outdir, 'wb') as file:
    pickle.dump(bright_dict_total, file)

bins = np.logspace(np.floor(np.log10(np.amin(np.array(bright_dict_total.values())[np.array(bright_dict_total.values()) > 0]))),
                   np.ceil(np.log10(np.amax(bright_dict_total.values()))), num=int(len(obslist) / 10) + 1)
plt.hist(bright_dict_total.values(), bins=bins, histtype='step')
plt.yscale('log')
plt.xscale('log')
plt.title('Total Brightness Histogram')
plt.xlabel('Integrated Brightness')
plt.ylabel('Counts')
plt.savefig('%s/total_brightness_hist.pdf' % (args.outdir))
plt.close()
