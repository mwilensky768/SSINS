from __future__ import absolute_import, division, print_function

from SSINS import INS, SS, version
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


def test_mask_to_flags():
    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    ss = SS()
    ss.read(testfile)

    ins = INS(ss)
    freq_inds_1 = np.arange(0, len(ins.freq_array), 2)
    freq_inds_2 = np.arange(1, len(ins.freq_array), 2)
    ins.metric_array[1, freq_inds_1] = np.ma.masked
    ins.metric_array[3, freq_inds_1] = np.ma.masked
    ins.metric_array[7, freq_inds_2] = np.ma.masked
    ins.metric_array[-2, freq_inds_2] = np.ma.masked
    flags = ins.mask_to_flags()

    test_flags = np.zeros(flags.shape, dtype=bool)
    test_flags[1:5, freq_inds_1] = True
    test_flags[7, freq_inds_2] = True
    test_flags[8, freq_inds_2] = True
    test_flags[-3:-1, freq_inds_2] = True

    assert np.all(flags == test_flags), "Test flags were not equal to calculated flags."


def test_write():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'
    prefix = os.path.join(DATA_PATH, '%s_test' % obs)
    data_outfile = '%s_SSINS_data.h5' % prefix
    z_score_outfile = '%s_SSINS_z_score.h5' % prefix
    flags_outfile = '%s_SSINS_flags.h5' % prefix
    mask_outfile = '%s_SSINS_mask.h5' % prefix
    match_outfile = '%s_SSINS_match_events.yml' % prefix

    ss = SS()
    ss.read(testfile, flag_choice='original')

    ins = INS(ss)
    # Mock some events
    ins.match_events.append((0, slice(1, 3), 'shape', 5))
    ins.match_events.append((1, slice(1, 3), 'shape', 5))
    ins.metric_array[:2, 1:3] = np.ma.masked
    ins.metric_ms = ins.mean_subtract()

    ins.write(prefix, output_type='data')
    ins.write(prefix, output_type='z_score')
    ins.write(prefix, output_type='flags')
    ins.write(prefix, output_type='mask')
    ins.write(prefix, output_type='match_events')
    with pytest.raises(ValueError):
        ins.write(prefix, output_type='bad_label')

    new_ins = INS(data_outfile, mask_file=mask_outfile, match_events_file=match_outfile)
    assert np.all(ins.metric_array == new_ins.metric_array), "Elements of the metric array were not equal"
    assert np.all(ins.weights_array == new_ins.weights_array), "Elements of the weights array were not equal"
    assert np.all(ins.metric_array.mask == new_ins.metric_array.mask), "Elements of the mask were not equal"
    assert np.all(ins.metric_ms == new_ins.metric_ms), "Elements of the metric_ms were not equal"
    assert np.all(ins.match_events == new_ins.match_events), "Elements of the match_events were not equal"

    for path in [data_outfile, z_score_outfile, flags_outfile, mask_outfile, match_outfile]:
        os.remove(path)


def test_write_mwaf():
    from astropy.io import fits

    obs = '1061313128_99bl_1pol_half_time_SSINS'
    testfile = os.path.join(DATA_PATH, '%s.h5' % obs)
    prefix = os.path.join(DATA_PATH, '%s_test' % obs)
    ins = INS(testfile)
    mwaf_files = [os.path.join(DATA_PATH, '1061313128_12.mwaf')]
    bad_mwaf_files = [os.path.join(DATA_PATH, 'bad_file_path')]

    # shape of that mwaf file
    ins.metric_array = np.ma.ones([223, 768, 1])
    ins.metric_array[100, 32 * 11: 32 * (11 + 1)] = np.ma.masked

    flags = ins.mask_to_flags()
    new_flags = np.repeat(flags[:, np.newaxis, 32 * 11: 32 * (11 + 1)], 8256, axis=1).reshape((224 * 8256, 32))

    # Test some defensive errors
    with pytest.raises(IOError):
        ins.write(prefix, output_type='mwaf', mwaf_files=bad_mwaf_files)
    with pytest.raises(ValueError):
        ins.write(prefix, output_type='mwaf', mwaf_files=mwaf_files,
                  mwaf_method='bad_method')
    with pytest.raises(ValueError):
        ins.write(prefix, output_type='mwaf', mwaf_files=None)

    ins.write('%s_add' % prefix, output_type='mwaf', mwaf_files=mwaf_files)
    ins.write('%s_replace' % prefix, output_type='mwaf', mwaf_files=mwaf_files,
              mwaf_method='replace')

    with fits.open(mwaf_files[0]) as old_mwaf_hdu:
        with fits.open('%s_add_12.mwaf' % prefix) as add_mwaf_hdu:
            assert np.all(add_mwaf_hdu[1].data['FLAGS'] == old_mwaf_hdu[1].data['FLAGS'] + new_flags)
    with fits.open('%s_replace_12.mwaf' % prefix) as replace_mwaf_hdu:
        assert np.all(replace_mwaf_hdu[1].data['FLAGS'] == new_flags)

    for path in ['%s_add_12.mwaf' % prefix, '%s_replace_12.mwaf' % prefix]:
        os.remove(path)
