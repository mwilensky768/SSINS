import argparse
from SSINS import util
from SSINS import INS
from SSINS import MF
from SSINS import Catalog_Plot as cp
from matplotlib import cm
import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)

parser = argparse.ArgumentParser()
parser.add_argument('basedir', help='The directory which contains the arrs, figs, and metadata subdirectories')
parser.add_argument('outdir', help='The base directory for the outputs')
parser.add_argument('obsfile', help='Path to a textfile which contains an obsid in each lines')
parser.add_argument('flag_choice', help='The flag choice for the saved data (None, custom, or original)')
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
obslist = [obslist[4 * i] for i in range(95)]

i = 0
for obs in obslist:
    i += 1
    print(i)
    read_paths = util.read_paths_construct(args.basedir, args.flag_choice, obs, 'INS')
    ins = INS(read_paths=read_paths, flag_choice=args.flag_choice, obs=obs[:-6],
              outpath=args.outdir, order=args.order)
    ins.data = np.ma.concatenate((ins.data, ) + tuple([np.load('%s/arrs/%s.%s.HH_%s_INS_data.npym' % (args.basedir, obs[:-6], pol, args.flag_choice)) for pol in ['yy', 'xy', 'yx']]), axis=3)
    ins.data.mask = np.zeros(ins.data.shape, dtype=bool)
    ins.Nbls = np.ma.concatenate((ins.Nbls, ) + tuple([np.load('%s/arrs/%s.%s.HH_%s_INS_Nbls.npym' % (args.basedir, obs[:-6], pol, args.flag_choice)) for pol in ['yy', 'xy', 'yx']]), axis=3)
    ins.Nbls.mask = np.zeros(ins.data.shape, dtype=bool)
    ins.pols = ['XX', 'YY', 'XY', 'YX']
    ins.data_ms = ins.mean_subtract(order=args.order)
    cp.INS_plot(ins, vmax=0.1, ms_vmin=-5, ms_vmax=5)
    ins.data[:, 0, :82] = np.ma.masked
    ins.data[:, 0, -50:] = np.ma.masked
    ins.data_ms = ins.mean_subtract(order=args.order)
    mf = MF(ins, sig_thresh=5, shape_dict=shape_dict, N_thresh=20)
    mf.apply_match_test(order=args.order, apply_N_thresh=True)
    cp.MF_plot(mf, vmax=0.1, ms_vmin=-5, ms_vmax=5)
