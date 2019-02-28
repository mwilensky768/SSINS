from __future__ import absolute_import, division, print_function

from SSINS import MF, INS, ES
from SSINS.data import DATA_PATH
import os
import numpy as np
import pytest


def test_init():

    freq_path = os.path.join(DATA_PATH, 'MWA_Highband_Freq_Array.npy')
    freqs = np.load(freq_path)

    # Make a shape that encompasses the first five channels
    ch_width = freqs[0, 1] - freqs[0, 0]
    shape = [freqs[0, 0] - 0.1 * ch_width, freqs[0, 4] + 0.1 * ch_width]
    shape_dict = {'shape': shape}

    mf_1 = MF(freqs, 5, shape_dict=shape_dict)

    assert mf_1.slice_dict['shape'] == slice(0, 5), "It did not set the shape correctly"
    assert mf_1.slice_dict['narrow'] is None, "narrow did not get set correctly"
    assert mf_1.slice_dict['streak'] == slice(None), "streak did not get set correctly"

    # Test disabling streak/narrow
    mf_2 = MF(freqs, 5, shape_dict=shape_dict, narrow=False, streak=False)

    assert 'narrow' not in mf_2.slice_dict, "narrow is still in the shape_dict"
    assert 'streak' not in mf_2.slice_dict, "streak still in shape_dict"
    assert 'shape' in mf_2.slice_dict, "shape not in shape_dict"

    # Test if error gets raised with bad shape_dict
    try:
        mf_3 = MF(freqs, 5, shape_dict={}, streak=False, narrow=False)
    except ValueError:
        pass


def test_match_test():

    obs = '1061313128_99bl_1pol_half_time'
    insfile = os.path.join(DATA_PATH, '%s_SSINS.uvh5' % obs)

    ins = INS(insfile)

    # Mock a simple metric_array and freq_array
    ins.metric_array = np.ma.ones([10, 20, 1])
    ins.weights_array = np.copy(ins.metric_array)
    ins.freq_array = np.zeros([1, 20])
    ins.freq_array[0, :] = np.arange(20)

    # Make a shape dictionary for a shape that will be injected later
    shape = [7.9, 12.1]
    shape_dict = {'shape': shape}
    mf = MF(ins.freq_array, 5, shape_dict=shape_dict)

    # Inject a shape, narrow, and streak event
    ins.metric_array[3, 5] = 10
    ins.metric_array[5] = 10
    ins.metric_array[7, 7:13] = 10
    ins.metric_ms = ins.mean_subtract()

    t_max, f_max, R_max, shape_max = mf.match_test(ins)
    print(shape_max)

    assert t_max == 5, "Wrong time"
    assert f_max == slice(None), "Wrong freq"
    assert shape_max == 'streak', "Wrong shape"


def test_apply_match_test():

    obs = '1061313128_99bl_1pol_half_time'
    insfile = os.path.join(DATA_PATH, '%s_SSINS.uvh5' % obs)

    ins = INS(insfile)

    # Mock a simple metric_array and freq_array
    ins.metric_array = np.ma.ones([10, 20, 1])
    ins.weights_array = np.copy(ins.metric_array)
    ins.freq_array = np.zeros([1, 20])
    ins.freq_array[0, :] = np.arange(20)

    # Make a shape dictionary for a shape that will be injected later
    shape = [7.9, 12.1]
    shape_dict = {'shape': shape}
    mf = MF(ins.freq_array, 5, shape_dict=shape_dict)

    # Inject a shape, narrow, and streak event
    ins.metric_array[3, 5] = 10
    ins.metric_array[5] = 10
    ins.metric_array[7, 7:13] = 10
    ins.metric_ms = ins.mean_subtract()

    es = mf.apply_match_test(ins, event_record=True)

    # Check that the right events are flagged
    test_mask = np.zeros(ins.metric_array.shape, dtype=bool)
    test_mask[3, 5] = 1
    test_mask[5] = 1
    test_mask[7, 7:13] = 1

    assert np.all(test_mask == ins.metric_array.mask), "Flags are incorrect"

    assert es.match_events == [(5, slice(None), 'streak'),
                               (7, slice(7, 13), 'shape'),
                               (3, slice(5, 6), 'narrow')], "Events are incorrect"


def test_samp_thresh():

    obs = '1061313128_99bl_1pol_half_time'
    insfile = os.path.join(DATA_PATH, '%s_SSINS.uvh5' % obs)

    ins = INS(insfile)

    # Mock a simple metric_array and freq_array
    ins.metric_array = np.ma.ones([10, 20, 1])
    ins.weights_array = np.copy(ins.metric_array)
    ins.freq_array = np.zeros([1, 20])
    ins.freq_array[0, :] = np.arange(20)

    # Arbitrarily flag enough data in channel 10
    mf = MF(ins.freq_array, 5, streak=False, N_samp_thresh=5)
    ins.metric_array[3:, 10] = np.ma.masked
    ins.metric_array[3:, 9] = np.ma.masked
    # Put in an outlier so it gets to samp_thresh_test
    ins.metric_array[0, 11] = 10
    ins.metric_ms = ins.mean_subtract()
    bool_ind = np.zeros(ins.metric_array.shape, dtype=bool)
    bool_ind[:, 10] = 1
    bool_ind[:, 9] = 1
    bool_ind[0, 11] = 1

    es = mf.apply_match_test(ins, event_record=True, apply_samp_thresh=True)

    # Test stuff
    assert np.all(ins.metric_array.mask == bool_ind), "The right flags were not applied"
    assert np.all(es.samp_thresh_events == np.array([9, 10])), "The events weren't appended correctly"
