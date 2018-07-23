import argparse
import INS
import numpy as np
import glob

parser = argparse.ArgumentParser()
parser.add_argument('inpath', action='store', help='The base directory')
args = parser.parse_args()

data = np.load('%s/arrs/')
