from __future__ import absolute_import, division, print_function

from SSINS import INS, SS
from SSINS.data import DATA_PATH
import numpy as np
import os
import pytest


def test_init():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    ss = SS()
    ss.read(testfile, flag_choice='original')

    ins = INS(ss)

    # Mock the averaging method
    new_shape = [ss.Ntimes, ss.Nbls, ss.Nfreqs, ss.Npols]
    test_dat = np.mean(np.abs(ss.data_array).reshape(new_shape), axis=1)

    # Mock the weights array
    test_weights = np.sum(np.logical_not(ss.data_array.mask).reshape(new_shape), axis=1)

    # Check that the data array averaged correctly
    assert np.all(test_dat == ins.metric_array), "Averaging did not work as intended."
    # Check that the weights summed correctly
    assert np.all(test_weights == ins.weights_array), "Weights did not sum properly"


def test_mean_subtract():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    ss = SS()
    ss.read(testfile)

    ins = INS(ss, order=0)

    old_dat = np.copy(ins.metric_ms)

    # Mask the first five frequencies and last two at the first and second times
    ins.metric_array[0, :5] = np.ma.masked

    # Calculate the new mean-subtracted spectrum only over the first few masked frequencies
    ins.metric_ms[:, :5] = ins.mean_subtract(f=slice(0, 5))

    # See if a new mean was calculated over the first five frequencies
    assert not np.all(old_dat[1:, :5] == ins.metric_ms[1:, :5]), "All elements of the ms array are still equal"


def test_polyfit():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    ss = SS()
    ss.read(testfile)

    ins = INS(ss, order=1)

    # Mock some data for which the polyfit is exact
    x = np.arange(1, 11)
    ins.metric_array = np.ma.masked_array([[3 * x + 5 for i in range(3)]])
    ins.metric_array.mask = np.zeros(ins.metric_array.shape, dtype=bool)
    ins.metric_array = np.swapaxes(ins.metric_array, 0, 2)
    ins.weights_array = np.ones(ins.metric_array.shape)
    ins.metric_ms = ins.mean_subtract()

    assert np.all(ins.metric_ms == np.zeros(ins.metric_ms.shape)), "The polyfit was not exact"


def test_write():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'
    outfile = os.path.join(DATA_PATH, '%s_SSINS_test.h5' % obs)

    ss = SS()
    ss.read(testfile, flag_choice='original')

    ins = INS(ss)
    ins.write(outfile)

    new_ins = INS(outfile)
    assert np.all(ins.metric_array == new_ins.metric_array), "Elements of the metric array were not equal"
    assert np.all(ins.weights_array == new_ins.weights_array), "Elements of the weights array were not equal"
    assert np.all(ins.metric_array.mask == new_ins.metric_array.mask), "Elements of the mask were not equal"

    os.remove(outfile)
