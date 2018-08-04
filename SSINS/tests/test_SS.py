from __future__ import absolute_import, division, print_function

import nose.tools as nt
from SSINS.data import DATA_PATH
from SSINS import SS
from SSINS import INS
from SSINS import VDH
from SSINS import ES
import shutil
import os
import numpy as np

"""
Tests the various capabilities of the sky_subtract class
"""


def test_ES_construct_write():
    """
    Read in uvfits file to SS class, construct usual data products, save outputs,
    read back in to assure saving went ok.
    """

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'
    outpath = os.path.join(DATA_PATH, 'test_outputs/')
    shape_dict = {'TV7': np.array([1.81e+08, 1.88e+08]),
                  'TV6': np.array([1.74e+08, 1.81e+08]),
                  'TV8': np.array([1.88e+08, 1.95e+08])}

    ss = SS(obs=obs, outpath=outpath, inpath=testfile)
    ss.ES_prepare()
    ss.save_data()

    read_paths = {}
    tags = ['match', 'chisq', 'samp_thresh']
    tag = ''
    for subtag in tags:
        if len(getattr(ss.INS, '%s_events' % (subtag))):
            tag += '_%s' % subtag
            rw_path = '%s/arrs/%s_%s_INS_%s_events.npy' %\
                      (outpath, obs, ss.INS.flag_choice, subtag)
            nt.ok_(os.path.exists(rw_path))
            read_paths['%s_events' % subtag] = rw_path
            if subtag is 'match' or subtag is 'chisq':
                rw_path = '%s/arrs/%s_%s_INS_%s_hists.npy' %\
                          (outpath, obs, ss.INS.flag_choice, subtag)
                nt.ok_(os.path.exists(rw_path))
                read_paths['%s_hists' % subtag] = rw_path
    for attr in ['data', 'data_ms', 'Nbls', 'counts', 'bins']:
        rw_path = '%s/arrs/%s_%s_INS_%s%s.npym' % \
                  (outpath, obs, ss.INS.flag_choice, attr, tag)
        nt.ok_(os.path.exists(rw_path))
        read_paths[attr] = rw_path
    for attr in ['freq_array', 'pols', 'vis_units']:
        rw_path = '%s/metadata/%s_%s.npy' % (outpath, obs, attr)
        nt.ok_(os.path.exists(rw_path))
        read_paths[attr] = rw_path

    test_INS = INS(obs=obs, outpath=outpath, read_paths=read_paths)
    for attr in ['pols', 'vis_units']:
        nt.ok_(np.all(getattr(test_INS, attr) == getattr(ss.INS, attr)))
    for attr in ['data', 'data_ms', 'Nbls', 'counts', 'bins', 'freq_array']:
        nt.ok_(np.allclose(getattr(test_INS, attr), getattr(ss.INS, attr), atol=1e-5))
    for attr in ['match_events', 'chisq_events']:
        nt.ok_(getattr(test_INS, attr) == getattr(ss.INS, attr))
    for attr in ['match_hists', 'chisq_hists']:
        print(attr)
        for i in range(len(getattr(ss.INS, attr))):
            for k in range(2):
                nt.ok_(np.all(getattr(test_INS, attr)[i][k] == getattr(ss.INS, attr)[i][k]))
    nt.ok_(np.all(test_INS.samp_thresh_events == ss.INS.samp_thresh_events))

    read_paths = {}
    for attr in ['counts', 'bins', 'fits', 'errors']:
        rw_path = '%s/arrs/%s_%s_VDH_%s.npy' % (outpath, obs, ss.VDH.flag_choice, attr)
        nt.ok_(os.path.exists(rw_path), msg='%s does not exist' % rw_path)
        read_paths[attr] = rw_path
    for attr in ['freq_array', 'pols', 'vis_units']:
        rw_path = '%s/metadata/%s_%s.npy' % (outpath, obs, attr)
        nt.ok_(os.path.exists(rw_path), msg='%s does not exist' % rw_path)
        read_paths[attr] = rw_path
    rw_path = '%s/arrs/%s_%s_VDH_MLEs.npym' % (outpath, obs, ss.VDH.flag_choice)
    nt.ok_(os.path.exists(rw_path))
    read_paths['MLEs'] = rw_path

    test_VDH = VDH(obs=obs, outpath=outpath, read_paths=read_paths)
    for attr in ['counts', 'bins', 'fits', 'errors']:
        for i in range(test_VDH.counts.shape[0]):
            for k in range(test_VDH.counts.shape[1]):
                nt.ok_(np.all(getattr(test_VDH, attr)[i, k] == getattr(ss.VDH, attr)[i, k]))
    for i in range(test_VDH.MLEs.shape[0]):
        print('MLEs')
        nt.ok_(np.all(test_VDH.MLEs == ss.VDH.MLEs))
    nt.ok_(np.all(test_VDH.freq_array == ss.VDH.freq_array))
    for attr in ['pols', 'vis_units']:
        nt.ok_(np.all(getattr(test_VDH, attr) == getattr(ss.VDH, attr)))

    read_paths = {}
    for attr in ['vis_units', 'pols', 'grid', 'Nfreqs']:
        rw_path = '%s/metadata/%s_%s.npy' % (outpath, obs, attr)
        nt.ok_(os.path.exists(rw_path))
        read_paths[attr] = rw_path
    for attr in ['counts', 'exp_counts', 'exp_error', 'bins', 'cutoffs']:
        rw_path = '%s/arrs/%s_None_%s.npy' % (outpath, obs, attr)
        nt.ok_(os.path.exists(rw_path))
        read_paths[attr] = rw_path
    for attr in ['avgs', 'uv_grid']:
        rw_path = '%s/arrs/%s_None_%s.npym' % (outpath, obs, attr)
        nt.ok_(os.path.exists(rw_path))
        read_paths[attr] = rw_path

    test_ES = ES(obs=obs, outpath=outpath, read_paths=read_paths)
    for attr in ['vis_units', 'pols', 'grid', 'Nfreqs']:
        nt.ok_(np.all(getattr(test_ES, attr) == getattr(ss.ES, attr)))
    for attr in ['counts', 'exp_counts', 'exp_error', 'bins', 'cutoffs', 'avgs', 'uv_grid']:
        for i in range(len(test_ES.counts)):
            nt.ok_(np.all(getattr(test_ES, attr) == getattr(ss.ES, attr)))

    shutil.rmtree(outpath)
