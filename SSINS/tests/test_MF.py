from __future__ import absolute_import, division, print_function

from SSINS import MF, INS, ES
from SSINS.data import DATA_PATH
import os
import numpy as np
import pytest


def INS_mock():

    obs = '1061313128_99bl_1pol_half_time'
    insfile = os.path.join(DATA_PATH, '%s_SSINS.uvh5' % obs)

    ins = INS(insfile)

    # Mock a simple metric_array and freq_array
    ins.metric_array = np.ma.ones([10, 20, 1])
    ins.weights_array = np.copy(ins.metric_array)
    ins.freq_array = np.zeros([1, 20])
    ins.freq_array[0, :] = np.arange(20)

    return(ins)


def test_init():

    freq_path = os.path.join(DATA_PATH, 'MWA_Highband_Freq_Array.npy')
    freqs = np.load(freq_path)

    # Make a shape that encompasses the first five channels
    ch_width = freqs[0, 1] - freqs[0, 0]
    shape = [freqs[0, 0] - 0.1 * ch_width, freqs[0, 4] + 0.1 * ch_width]
    shape_dict = {'shape': shape}

    mf_1 = MF(freqs, 5, shape_dict=shape_dict)

    assert mf_1.slice_dict['shape'] == slice(0, 5), "It did not set the shape correctly"
    assert mf_1.slice_dict['point'] is None, "Point did not get set correctly"
    assert mf_1.slice_dict['streak'] == slice(None), "Streak did not get set correctly"

    # Test disabling streak/point
    mf_2 = MF(freqs, 5, shape_dict=shape_dict, point=False, streak=False)

    assert 'point' not in mf_2.slice_dict, "Point is still in the shape_dict"
    assert 'streak' not in mf_2.slice_dict, "Streak still in shape_dict"
    assert 'shape' in mf_2.slice_dict, "Shape not in shape_dict"

    # Test if error gets raised with bad shape_dict
    try:
        mf_3 = MF(freqs, 5, shape_dict={}, streak=False, point=False)
    except ValueError:
        pass


def test_match_test():

    ins = INS_mock()

    # Make a shape dictionary for a shape that will be injected later
    shape = [7.9, 12.1]
    shape_dict = {'shape': shape}
    mf = MF(ins.freq_array, 5, shape_dict=shape_dict)

    # Inject a shape, point, and streak event
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

    ins = INS_mock()

    # Make a shape dictionary for a shape that will be injected later
    shape = [7.9, 12.1]
    shape_dict = {'shape': shape}
    mf = MF(ins.freq_array, 5, shape_dict=shape_dict)

    # Inject a shape, point, and streak event
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
                               (3, slice(5, 6), 'point')], "Events are incorrect"
