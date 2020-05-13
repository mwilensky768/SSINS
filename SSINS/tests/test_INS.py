from SSINS import INS, SS, version
from SSINS.data import DATA_PATH
import numpy as np
import os
import pytest
from pyuvdata import UVData, UVFlag


def test_init():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    ss = SS()
    ss.read(testfile, flag_choice='original', diff=True)
    # Needs to be in time order for averaging comparison to work
    ss.reorder_blts(order='time')

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
    ss.read(testfile, diff=True)

    ins = INS(ss, order=0)

    old_dat = np.copy(ins.metric_ms)

    # Mask the first five frequencies and last two at the first and second times
    ins.metric_array[0, :5] = np.ma.masked

    # Calculate the new mean-subtracted spectrum only over the first few masked frequencies
    ins.metric_ms[:, :5] = ins.mean_subtract(freq_slice=slice(0, 5))

    # See if a new mean was calculated over the first five frequencies
    assert not np.all(old_dat[1:, :5] == ins.metric_ms[1:, :5]), "All elements of the ms array are still equal"


def test_polyfit():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    ss = SS()
    ss.read(testfile, diff=True)

    ins = INS(ss, order=1)

    # Mock some data for which the polyfit is exact
    x = np.arange(1, 11)
    ins.metric_array = np.ma.masked_array([[3 * x + 5 for i in range(3)]])
    ins.metric_array.mask = np.zeros(ins.metric_array.shape, dtype=bool)
    ins.metric_array = np.swapaxes(ins.metric_array, 0, 2)
    ins.weights_array = np.ones(ins.metric_array.shape)
    ins.metric_ms, coeffs = ins.mean_subtract(return_coeffs=True)
    test_coeffs = np.zeros((ins.order + 1, ) + ins.metric_ms.shape[1:])
    test_coeffs[0, :] = 3
    test_coeffs[1, :] = 5

    assert np.all(ins.metric_ms == np.zeros(ins.metric_ms.shape)), "The polyfit was not exact"
    assert np.all(np.allclose(coeffs, test_coeffs)), "The polyfit got the wrong coefficients"

    ins.metric_array[:] = np.ma.masked
    ins.metric_ms = ins.mean_subtract()
    assert np.all(ins.metric_ms.mask), "The metric_ms array was not all masked"


def test_mask_to_flags():
    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'
    prefix = os.path.join(DATA_PATH, '%s_test' % obs)
    flags_outfile = '%s_SSINS_flags.h5' % prefix

    ss = SS()
    ss.read(testfile, diff=True)

    uvd = UVData()
    uvd.read(testfile)

    uvf = UVFlag(uvd, mode='flag', waterfall=True)
    # start with some flags so that we can test the intended OR operation
    uvf.flag_array[6, :] = True
    ins = INS(ss)

    # Check error handling
    with pytest.raises(ValueError):
        bad_uvf = UVFlag(uvd, mode='metric', waterfall=True)
        err_uvf = ins.flag_uvf(uvf=bad_uvf)
    with pytest.raises(ValueError):
        bad_uvf = UVFlag(uvd, mode='flag', waterfall=False)
        err_uvf = ins.flag_uvf(uvf=bad_uvf)
    with pytest.raises(ValueError):
        bad_uvf = UVFlag(uvd, mode='flag', waterfall=True)
        # Pretend the data is off by 1 day
        bad_uvf.time_array += 1
        err_uvf = ins.flag_uvf(uvf=bad_uvf)

    # Pretend we flagged the INS object
    freq_inds_1 = np.arange(0, len(ins.freq_array), 2)
    freq_inds_2 = np.arange(1, len(ins.freq_array), 2)
    ins.metric_array[1, freq_inds_1] = np.ma.masked
    ins.metric_array[3, freq_inds_1] = np.ma.masked
    ins.metric_array[7, freq_inds_2] = np.ma.masked
    ins.metric_array[-2, freq_inds_2] = np.ma.masked

    # Make a NEW uvflag object
    new_uvf = ins.flag_uvf(uvf=uvf, inplace=False)

    # Construct the expected flags by hand
    test_flags = np.zeros_like(new_uvf.flag_array)
    test_flags[1:5, freq_inds_1] = True
    test_flags[6, :] = True
    test_flags[7, freq_inds_2] = True
    test_flags[8, freq_inds_2] = True
    test_flags[-3:-1, freq_inds_2] = True

    # Check that new flags are correct
    assert np.all(new_uvf.flag_array == test_flags), "Test flags were not equal to calculated flags."
    # Check that the input uvf was not edited in place
    assert new_uvf != uvf, "The UVflag object was edited inplace and should not have been."

    # Edit the uvf inplace
    inplace_uvf = ins.flag_uvf(uvf=uvf, inplace=True)

    # Check that new flags are correct
    assert np.all(inplace_uvf.flag_array == test_flags), "Test flags were not equal to calculated flags."
    # Check that the input uvf was not edited in place
    assert inplace_uvf == uvf, "The UVflag object was not edited inplace and should have been."

    # Test write/read
    ins.write(prefix, output_type='flags', uvf=uvf)
    read_uvf = UVFlag(flags_outfile, mode='flag', waterfall=True)
    # Check equality
    assert read_uvf == uvf, "UVFlag objsect differs after read"

    os.remove(flags_outfile)


def test_write():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'
    prefix = os.path.join(DATA_PATH, '%s_test' % obs)
    data_outfile = '%s_SSINS_data.h5' % prefix
    z_score_outfile = '%s_SSINS_z_score.h5' % prefix
    mask_outfile = '%s_SSINS_mask.h5' % prefix
    match_outfile = '%s_SSINS_match_events.yml' % prefix
    sep_data_outfile = '%s.SSINS.data.h5' % prefix

    ss = SS()
    ss.read(testfile, flag_choice='original', diff=True)

    ins = INS(ss)
    # Mock some events
    ins.match_events.append((0, slice(1, 3), 'shape', 5))
    ins.match_events.append((1, slice(1, 3), 'shape', 5))
    ins.metric_array[:2, 1:3] = np.ma.masked
    ins.metric_ms = ins.mean_subtract()

    ins.write(prefix, output_type='data')
    ins.write(prefix, output_type='z_score')
    ins.write(prefix, output_type='mask')
    ins.write(prefix, output_type='match_events')
    ins.write(prefix, output_type='data', sep='.')
    with pytest.raises(ValueError):
        ins.write(prefix, output_type='bad_label')
    with pytest.raises(ValueError):
        ins.write(prefix, output_type='flags')

    new_ins = INS(data_outfile, mask_file=mask_outfile, match_events_file=match_outfile)
    assert np.all(ins.metric_array == new_ins.metric_array), "Elements of the metric array were not equal"
    assert np.all(ins.weights_array == new_ins.weights_array), "Elements of the weights array were not equal"
    assert np.all(ins.metric_array.mask == new_ins.metric_array.mask), "Elements of the mask were not equal"
    assert np.all(ins.metric_ms == new_ins.metric_ms), "Elements of the metric_ms were not equal"
    assert np.all(ins.match_events == new_ins.match_events), "Elements of the match_events were not equal"
    assert os.path.exists(sep_data_outfile), "sep_data_outfile was note written"

    for path in [data_outfile, z_score_outfile, mask_outfile, match_outfile,
                 sep_data_outfile]:
        os.remove(path)


def test_write_mwaf():
    from astropy.io import fits

    obs = '1061313128_99bl_1pol_half_time_SSINS'
    testfile = os.path.join(DATA_PATH, '%s.h5' % obs)
    prefix = os.path.join(DATA_PATH, '%s_test' % obs)
    ins = INS(testfile)
    mwaf_files = [os.path.join(DATA_PATH, '1061313128_12.mwaf')]
    bad_mwaf_files = [os.path.join(DATA_PATH, 'bad_file_path')]

    # Compatible shape with mwaf file
    ins.metric_array = np.ma.ones([55, 384, 1])
    ins.metric_array[50, 16 * 11: 16 * (11 + 1)] = np.ma.masked

    # metadata from the input file, hardcoded for testing purposes
    time_div = 4
    freq_div = 2
    NCHANS = 32
    boxint = 11
    Nbls = 8256
    NSCANS = 224
    flags = ins.mask_to_flags()
    # Repeat in time
    time_rep_flags = np.repeat(flags, time_div, axis=0)
    # Repeat in freq
    freq_time_rep_flags = np.repeat(time_rep_flags, freq_div, axis=1)
    # Repeat in bls
    freq_time_bls_rep_flags = np.repeat(freq_time_rep_flags[:, np.newaxis, NCHANS * boxint: NCHANS * (boxint + 1)], Nbls, axis=1)
    # This shape is on MWA wiki. Reshape to this shape.
    new_flags = freq_time_bls_rep_flags.reshape((NSCANS * Nbls, NCHANS))

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


def test_select():

    obs = '1061313128_99bl_1pol_half_time_SSINS'
    testfile = os.path.join(DATA_PATH, '%s.h5' % obs)
    ins = INS(testfile)

    Ntimes = len(ins.time_array)
    ins.select(times=ins.time_array[3:-3], freq_chans=np.arange(24))

    assert ins.metric_array.shape[0] == Ntimes - 6
    assert ins.metric_array.shape[1] == 24
    for param in ins._data_params:
        assert getattr(ins, param).shape == ins.metric_array.shape


def test_data_params():
    obs = '1061313128_99bl_1pol_half_time_SSINS'
    testfile = os.path.join(DATA_PATH, '%s.h5' % obs)
    ins = INS(testfile)

    assert ins._data_params == ['metric_array', 'weights_array', 'metric_ms', 'sig_array']


def test_spectrum_type_file_init():
    obs = "1061313128_99bl_1pol_half_time_SSINS"
    testfile = os.path.join(DATA_PATH, f"{obs}.h5")
    ins = INS(testfile)

    assert ins.spectrum_type == "cross"

    with pytest.raises(ValueError, match="Reading in a 'cross' spectrum as 'auto'."):
        ins = INS(testfile, spectrum_type="auto")


def test_spectrum_type_bl_init():
    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, f'{obs}.uvfits')
    file_type = 'uvfits'

    ss = SS()
    ss.read(testfile, diff=True)

    ins = INS(ss)
    assert "Initialized spectrum_type:cross from visibility data." in ins.history

    with pytest.raises(ValueError, match="Requested spectrum type is 'auto', but no autos exist."):
        ins = INS(ss, spectrum_type="auto")
