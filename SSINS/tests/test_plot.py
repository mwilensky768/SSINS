from __future__ import absolute_import, division, print_function

from SSINS import Catalog_Plot as cp
from SSINS import SS, INS, MF
from SSINS.data import DATA_PATH
import numpy as np
import os


def test_INS_plot():

    obs = '1061313128_99bl_1pol_half_time'
    insfile = os.path.join(DATA_PATH, '%s_SSINS.h5' % obs)
    outdir = os.path.join(DATA_PATH, 'test_plots')
    prefix = '%s/%s' % (outdir, obs)
    outfile = '%s_SSINS.pdf' % prefix

    ins = INS(insfile)

    cp.INS_plot(ins, prefix)

    assert os.path.exists(outfile), "The plot was not made"

    os.remove(outfile)
    os.rmdir(outdir)


def test_VDH_plot():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    outdir = os.path.join(DATA_PATH, 'test_plots')
    prefix = '%s/%s' % (outdir, obs)
    outfile = '%s_VDH.pdf' % prefix

    ss = SS()
    ss.read(testfile, flag_choice='original')

    cp.VDH_plot(ss, prefix)

    assert os.path.exists(outfile), "The plot was not made"

    os.remove(outfile)
    os.rmdir(outdir)
