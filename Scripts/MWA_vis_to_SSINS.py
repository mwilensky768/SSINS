from SSINS import INS, SS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filelist', nargs='*', help='List of gpubox files.')
parser.add_argument('-d', '--outdir', help='The output directory of the h5 file')
parser.add_argument('-o', '--obsid', help='The obsid of the files.')
args = parser.parse_args()

ss = SS()
ss.read(args.filelist, correct_cable_len=True, phase_to_pointing_center=True, ant_str='cross')

ins = INS(ss)
ins.history += "Read in vis data: applied cable corrections and phased to pointing center. "

prefix = '%s/%s' % (args.outdir, args.obsid)
ins.write(prefix)
