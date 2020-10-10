import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyuvdata import UVFlag

"""
A script for comparing masks from the same obsid with different settings.
"""

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--obsid", help="The obsid", required=True)
parser.add_argument("-a", "--file_a", help="The first mask to use for comparison",
                    required=True)
parser.add_argument("-b", "--file_b", help="The second mask to use for comparison",
                    required=True)
parser.add_argument("-x", "--label_a",
                    help="The label for file_a. Should identify settings in some way. Will go in output file name.",
                    required=True)
parser.add_argument("-y", "--label_b",
                    help="The label for file_b. Should identify settings in some way. Will go in output file name.",
                    required=True)
parser.add_argument("d", "--data_file", help="The SSINS data file", required=True)
parser.add_argument("-p", "--outdir", help="The output directory", required=True)

uvfa = UVFlag(args.file_a)
uvfb = UVFlag(args.file_b)

and_arr = np.logical_and(uvfa.flag_array, uvfb.flag_array)
a_not_b = np.logical_and(uvfa.flag_array, np.logical_not(uvfb.flag_array))
b_not_a = np.logical_and(uvfb.flag_array, np.logical_not(uvfa.flag_array))
neither = np.logical_and(np.logical_not(uvfa.flag_array), np.logical_not(uvfb.flag_array))

fig, ax = plt.figure(figsize=(16, 12))
# Do not have to use neither with this initialization
flag_table = np.ones_like(uvfa.flag_array[:, :, 0]).astype(float)
flag_table[and_arr] = 7
flag_table[a_not_b] = 5
flag_table[b_not_a] = 3

# Prepare a colormap.
cmap = plt.cm.colors.ListedColormap(
    ["slategray", "darkturquoise", "plum", "lemonchiffon"]
)
bounds = [0, 2, 4, 6, 8]
norm = colors.BoundaryNorm(bounds, cmap.N)


cax = ax.imshow(flag_table, aspect='auto',
                extent=[uvf.freq_array[0] / 1e6, uvf.freq_array[-1] / 1e6,
                        uvf.time_array[-1] - JD, uvf.time_array[0] - JD],
                cmap=cmap,
                vmin=0, vmax=8)
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel(f'JD - {JD}')
ax.set_yticklabels([Time(ytick, format='jd').iso[:-4] for ytick in ax.get_yticks()])
ax.set_title(f"{args.obsid} {args.label_a} vs. {args.label_b}")

cbar_ticklabels = ["Flagged in Neither", f"Flagged in {args.label_b}",
                   f"Flagged in {args.label_a}", "Flagged in Both"]

# Configure the colorbar so that labels are at the center of each section.
cbar = fig.colorbar(cax)
cbar_ticks = np.arange(1, 9, 2)
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(cbar_ticklabels)

fig.savefig(f"{args.outdir}/{args.obsid}_{args.label_a}_{args.label_b}_SSINS_flag_comparison.pdf")
