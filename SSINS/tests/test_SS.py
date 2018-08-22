from __future__ import absolute_import, division, print_function

import nose.tools as nt
from SSINS.data import DATA_PATH
from SSINS import SS
from SSINS import INS
from SSINS import VDH
from SSINS import ES
from SSINS import Catalog_Plot as cp
from SSINS import plot_lib as pl
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt

"""
Tests the various capabilities of the sky_subtract class
"""


def test_INS_construct_plot():
    """
    Construct an INS using the flags belonging to the uvfits file. Plot.
    """
    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    outpath = os.path.join(DATA_PATH, 'test_outputs/')
    figpath = os.path.join(DATA_PATH, 'figs/')
    file_type = 'uvfits'
    flag_choice = 'original'
    N_thresh = 15
    shape_dict = {'TV6': [1.74e8, 1.81e8],
                  'TV7': [1.81e8, 1.88e8],
                  'TV8': [1.88e8, 1.95e8]}

    ss = SS(obs=obs, outpath=outpath, inpath=testfile, flag_choice=flag_choice)
    ss.INS_prepare()
    ss.save_data()

    read_paths = {}
    tags = ['match', 'chisq', 'samp_thresh']
    tag = ''
    for subtag in tags:
        if len(getattr(ss.INS, '%s_events' % (subtag))):
            tag += '_%s' % subtag
            rw_path = '%s/arrs/%s_%s_INS_%s_events.npy' %\
                      (DATA_PATH, obs, ss.INS.flag_choice, subtag)
            read_paths['%s_events' % subtag] = rw_path
            if subtag is 'match' or subtag is 'chisq':
                rw_path = '%s/arrs/%s_%s_INS_%s_hists.npy' %\
                          (DATA_PATH, obs, ss.INS.flag_choice, subtag)
                nt.ok_(os.path.exists(rw_path))
                read_paths['%s_hists' % subtag] = rw_path
    for attr in ['data', 'data_ms', 'Nbls']:
        rw_path = '%s/arrs/%s_%s_INS_%s%s.npym' % \
                  (DATA_PATH, obs, ss.INS.flag_choice, attr, tag)
        read_paths[attr] = rw_path
    for attr in ['counts', 'bins']:
        rw_path = '%s/arrs/%s_%s_INS_%s%s.npy' % \
            (DATA_PATH, obs, ss.INS.flag_choice, attr, tag)
    for attr in ['freq_array', 'pols', 'vis_units']:
        rw_path = '%s/metadata/%s_%s.npy' % (DATA_PATH, obs, attr)
        read_paths[attr] = rw_path

    test_INS = INS(obs=obs, outpath=outpath, read_paths=read_paths)
    for attr in ['pols', 'vis_units']:
        nt.ok_(np.all(getattr(test_INS, attr) == getattr(ss.INS, attr)))
    for attr in ['data', 'data_ms', 'Nbls', 'counts', 'bins', 'freq_array']:
        nt.ok_(np.allclose(getattr(test_INS, attr), getattr(ss.INS, attr), atol=1e-5))
    for attr in ['match_events', 'chisq_events']:
        for i in range(len(getattr(ss.INS, attr))):
            nt.ok_(np.all(getattr(test_INS, attr)[i] == getattr(ss.INS, attr)[i]))
    for attr in ['match_hists', 'chisq_hists']:
        for i in range(len(getattr(ss.INS, attr))):
            for k in range(2):
                nt.ok_(np.all(getattr(test_INS, attr)[i][k] == getattr(ss.INS, attr)[i][k]))
    nt.ok_(np.all(test_INS.samp_thresh_events == ss.INS.samp_thresh_events))

    cp.INS_plot(ss.INS)
    # Test that the plot saved
    tags = ['match', 'chisq', 'samp_thresh']
    tag = ''
    for subtag in tags:
        if len(getattr(ss.INS, '%s_events' % (subtag))):
            tag += '_%s' % subtag
    nt.ok_(os.path.exists('%s/figs/%s_%s_INS_data%s.png' %
                          (ss.INS.outpath, ss.INS.obs, ss.INS.flag_choice, tag)))
    # Copy it for inspection
    shutil.copy('%s/figs/%s_%s_INS_data%s.png' %
                (ss.INS.outpath, ss.INS.obs, ss.INS.flag_choice, tag),
                '%s' % figpath)

    ss.MF_prepare(tests=('match', 'chisq', 'samp_thresh'), N_thresh=N_thresh,
                  shape_dict=shape_dict)
    ss.save_data()

    read_paths = {}
    tags = ['match', 'chisq', 'samp_thresh']
    tag = ''
    for subtag in tags:
        if len(getattr(ss.INS, '%s_events' % (subtag))):
            tag += '_%s' % subtag
            rw_path = '%s/arrs/%s_%s_INS_%s_events.npy' %\
                      (DATA_PATH, obs, ss.INS.flag_choice, subtag)
            nt.ok_(os.path.exists(rw_path))
            read_paths['%s_events' % subtag] = rw_path
            if subtag is 'match' or subtag is 'chisq':
                rw_path = '%s/arrs/%s_%s_INS_%s_hists.npy' %\
                          (DATA_PATH, obs, ss.INS.flag_choice, subtag)
                nt.ok_(os.path.exists(rw_path))
                read_paths['%s_hists' % subtag] = rw_path
    for attr in ['data', 'data_ms', 'Nbls']:
        rw_path = '%s/arrs/%s_%s_INS_%s%s.npym' % \
                  (DATA_PATH, obs, ss.INS.flag_choice, attr, tag)
        nt.ok_(os.path.exists(rw_path))
        read_paths[attr] = rw_path
    for attr in ['counts', 'bins']:
        rw_path = '%s/arrs/%s_%s_INS_%s%s.npy' % \
            (DATA_PATH, obs, ss.INS.flag_choice, attr, tag)
        nt.ok_(os.path.exists(rw_path))
    for attr in ['freq_array', 'pols', 'vis_units']:
        rw_path = '%s/metadata/%s_%s.npy' % (DATA_PATH, obs, attr)
        nt.ok_(os.path.exists(rw_path))
        read_paths[attr] = rw_path

    test_INS = INS(obs=obs, outpath=outpath, read_paths=read_paths)
    for attr in ['pols', 'vis_units']:
        nt.ok_(np.all(getattr(test_INS, attr) == getattr(ss.INS, attr)))
    for attr in ['data', 'data_ms', 'Nbls', 'counts', 'bins', 'freq_array']:
        nt.ok_(np.allclose(getattr(test_INS, attr), getattr(ss.INS, attr), atol=1e-5))
    for attr in ['match_events', 'chisq_events']:
        for i in range(len(getattr(ss.INS, attr))):
            nt.ok_(np.all(getattr(test_INS, attr)[i] == getattr(ss.INS, attr)[i]))
    for attr in ['match_hists', 'chisq_hists']:
        for i in range(len(getattr(ss.INS, attr))):
            for k in range(2):
                nt.ok_(np.all(getattr(test_INS, attr)[i][k] == getattr(ss.INS, attr)[i][k]))
    nt.ok_(np.all(test_INS.samp_thresh_events == ss.INS.samp_thresh_events))
    cp.MF_plot(ss.MF)

    tags = ['match', 'chisq', 'samp_thresh']
    tag = ''
    for subtag in tags:
        if len(getattr(ss.INS, '%s_events' % (subtag))):
            tag += '_%s' % subtag
    nt.ok_(os.path.exists('%s/figs/%s_%s_INS_data%s.png' %
                          (ss.INS.outpath, ss.INS.obs, ss.INS.flag_choice, tag)))
    # Copy it for inspection
    shutil.copy('%s/figs/%s_%s_INS_data%s.png' %
                (ss.INS.outpath, ss.INS.obs, ss.INS.flag_choice, tag),
                '%s' % figpath)

    # shutil.rmtree(outpath)


def test_VDH_construct_plot():
    """
    Construct a VDH with waterfall, plot it
    """

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    outpath = os.path.join(DATA_PATH, 'test_outputs/')
    figpath = os.path.join(DATA_PATH, 'figs/')
    file_type = 'uvfits'
    flag_choice = 'original'
    fit_tags = ['All', 'Flags']
    window = (1e3, 1e5)

    ss = SS(obs=obs, outpath=outpath, inpath=testfile, flag_choice=flag_choice)
    ss.VDH_prepare(fit_hist=True)
    ss.save_data()

    read_paths = {}
    for attr in ['counts', 'bins', 'fits', 'errors']:
        rw_path = '%s/arrs/%s_%s_VDH_%s.npy' % (DATA_PATH, obs, ss.VDH.flag_choice, attr)
        read_paths[attr] = rw_path
    for attr in ['freq_array', 'pols', 'vis_units']:
        rw_path = '%s/metadata/%s_%s.npy' % (DATA_PATH, obs, attr)
        read_paths[attr] = rw_path
    for attr in ['W_hist', 'MLEs']:
        rw_path = '%s/arrs/%s_%s_VDH_%s.npym' % (DATA_PATH, obs, ss.VDH.flag_choice, attr)
    read_paths['MLEs'] = rw_path

    test_VDH = VDH(obs=obs, outpath=outpath, read_paths=read_paths)
    for attr in ['counts', 'bins', 'fits', 'errors']:
        for i in range(len(test_VDH.counts)):
            nt.ok_(np.all(getattr(test_VDH, attr)[i] == getattr(ss.VDH, attr)[i]))
    nt.ok_(np.all(test_VDH.MLEs == ss.VDH.MLEs))
    nt.ok_(np.all(test_VDH.freq_array == ss.VDH.freq_array))
    for attr in ['pols', 'vis_units']:
        nt.ok_(np.all(getattr(test_VDH, attr) == getattr(ss.VDH, attr)))

    ss.VDH.rev_ind(ss.UV.data_array, window)
    cp.VDH_plot(ss.VDH)
    for i in range(1 + bool(ss.VDH.flag_choice)):
        nt.ok_(os.path.exists('%s/figs/%s_%s_VDH.png' %
                              (ss.VDH.outpath, obs, fit_tags[i])))
        shutil.copy('%s/figs/%s_%s_VDH.png' %
                    (ss.VDH.outpath, obs, fit_tags[i]), '%s' % figpath)
    # shutil.rmtree(outpath)


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
    MC_iter = int(1e2)

    ss = SS(obs=obs, outpath=outpath, inpath=testfile)
    ss.ES_prepare(MC_iter=MC_iter, shape_dict=shape_dict)
    ss.save_data()

    cp.ES_plot(ss.ES)

    for i, event in enumerate(ss.ES.events):
        nt.ok_(os.path.exists('%s/figs/%s_hist_%i.png' % (ss.ES.outpath, ss.ES.obs, i)))
        nt.ok_(os.path.exists('%s/figs/%s_grid_%i.png' % (ss.ES.outpath, ss.ES.obs, i)))

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
    for attr in ['data', 'data_ms', 'Nbls']:
        rw_path = '%s/arrs/%s_%s_INS_%s%s.npym' % \
                  (outpath, obs, ss.INS.flag_choice, attr, tag)
        nt.ok_(os.path.exists(rw_path))
        read_paths[attr] = rw_path
    for attr in ['counts', 'bins']:
        rw_path = '%s/arrs/%s_%s_INS_%s%s.npy' % \
            (outpath, obs, ss.INS.flag_choice, attr, tag)
        nt.ok_(os.path.exists(rw_path))
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
        for i in range(len(getattr(ss.INS, attr))):
            nt.ok_(np.all(getattr(test_INS, attr)[i] == getattr(ss.INS, attr)[i]))
    for attr in ['match_hists', 'chisq_hists']:
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
        for i in range(len(test_VDH.counts)):
            nt.ok_(np.all(getattr(test_VDH, attr)[i] == getattr(ss.VDH, attr)[i]))
    nt.ok_(np.all(test_VDH.MLEs == ss.VDH.MLEs))
    nt.ok_(np.all(test_VDH.freq_array == ss.VDH.freq_array))
    for attr in ['pols', 'vis_units']:
        nt.ok_(np.all(getattr(test_VDH, attr) == getattr(ss.VDH, attr)))

    read_paths = {}
    for attr in ['vis_units', 'pols', 'grid', 'freq_array']:
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
    for attr in ['vis_units', 'pols', 'grid', 'freq_array']:
        nt.ok_(np.all(getattr(test_ES, attr) == getattr(ss.ES, attr)))
    for attr in ['counts', 'exp_counts', 'exp_error', 'bins', 'cutoffs', 'avgs', 'uv_grid']:
        for i in range(len(test_ES.counts)):
            nt.ok_(np.all(getattr(test_ES, attr)[i] == getattr(ss.ES, attr)[i]))

    # shutil.rmtree(outpath)


def test_scatter():

    figpath = os.path.join(DATA_PATH, 'figs/')
    data = np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0], [0, 1]]), 1000)
    np.save('%s/rand_gauss_multivariate.npy' % (DATA_PATH), data)
    fig, ax = plt.subplots()
    pl.scatter_plot_2d(fig, ax, data[:, 0], data[:, 1])
    fig.savefig('%s/scatter_test.png' % figpath)
