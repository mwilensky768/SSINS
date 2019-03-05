from __future__ import absolute_import, division, print_function

from SSINS import Catalog_Plot as cp
from SSINS import SS, INS, ES, MF
from SSINS.data import DATA_PATH
import numpy as np
import os


def test_INS_plot():

    obs = '1061313128_99bl_1pol_half_time'
    insfile = os.path.join(DATA_PATH, '%s_SSINS.uvh5' % obs)
    outfile = os.path.join(DATA_PATH, '%s_SSINS.png' % obs)

    ins = INS(insfile)

    cp.INS_plot(ins, outfile)

    assert os.path.exists(outfile), "The plot was not made"

    os.remove(outfile)


def test_VDH_plot():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'
    outfile = os.path.join(DATA_PATH, '%s_VDH.png' % obs)

    ss = SS()
    ss.read(testfile, flag_choice='original')

    cp.VDH_plot(ss, outfile)

    assert os.path.exists(outfile), "The plot was not made"

    os.remove(outfile)
