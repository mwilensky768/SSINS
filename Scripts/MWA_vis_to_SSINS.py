from SSINS import INS, SS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filelist', nargs='*', help='List of gpubox files.')
parser.add_argument('-d', '--outdir', help='The output directory of the h5 file')
parser.add_argument('-o', '--obsid', help='The obsid of the files.')
args = parser.parse_args()

ss = SS()
ss.read(args.filelist, ant_str='cross')

ins = INS(ss)

prefix = '%s/%s' % (args.outdir, args.obsid)
ins.write(prefix)
