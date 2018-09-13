import argparse
from SSINS import util
from SSINS import INS
from SSINS import MF
from SSINS import Catalog_Plot as cp
from matplotlib import cm

parser = argparse.ArgumentParser()
parser.add_argument('basedir', help='The directory which contains the arrs, figs, and metadata subdirectories')
parser.add_argument('outdir', help='The base directory for the outputs')
parser.add_argument('obsfile', help='Path to a textfile which contains an obsid in each lines')
parser.add_argument('flag_choice' help='The flag choice for the saved data (None, custom, or original)')
parser.add_argument('--order', type=int, default=0,
                    help='The order for the polynomial fit on the mean-subtracted data. Use this if you see breathing modes that do not look like RFI. Default is 0, which just calculates the mean.')
parser.add_argument('--labels', nargs='*', help='Labels for the match filter shape.')
parser.add_argument('--mins', nargs='*', type=float, help='The minimum frequencies for the corresponding shapes in --labels.')
parser.add_argument('--maxs', nargs='*', type=float, help='The maximum frequencies for the corresponding shapes in --labels.')
args = parser.parse_args()

if args.labels is not None:
    shape_dict = {label: [min, max] for (label, min, max) in zip(args.labels, args.mins, args.maxs)}
else:
    shape_dict = {}

obslist = util.make_obslist(args.obsfile)

for obs in obslist:
    read_paths = util.read_paths_INS(args.basedir, args.flag_choice, obs)
    ins = INS(read_paths=read_paths, flag_choice=args.flag_choice, obs=obs,
              outpath=args.outdir, order=args.order)
    cp.INS_plot(ins)
    mf = MF(ins, sig_thresh=5, shape_dict=shape_dict)
    mf.apply_match_test(order=args.order)
    cp.MF_plot(mf)
