from SSINS import INS, SS
from SSINS.data import DATA_PATH
import numpy as np
import os
import pytest
from pyuvdata import UVData, UVFlag
from datetime import datetime


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:invalid value",
                            "ignore:SS.read")
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
    # Weights are floating-point, which introdices itty bitty errors compared to masked average.
    assert np.all(np.isclose(test_dat, ins.metric_array, rtol=1e-6, atol=1e-7)), "Averaging did not work as intended."
    # Check that the weights summed correctly
    assert np.all(test_weights == ins.weights_array), "Weights did not sum properly"

def test_extra_keywords():
    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    ss = SS()
    ss.read(testfile, flag_choice='original', diff=True)
    ins = INS(ss)

    assert ss.extra_keywords['dif_time'] is True
    assert ins.extra_keywords['dif_time'] is True

    ss.read(testfile, flag_choice='original', diff=False)
    ins = INS(ss)

    assert ss.extra_keywords['dif_time'] is False
    assert ins.extra_keywords['dif_time'] is False



def test_no_diff_start():
    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    # Don't diff - will fail to mask data array
    ss = SS()
    with pytest.warns(UserWarning, match="flag_choice will be ignored"):
        ss.read(testfile, flag_choice='original', diff=False)

    with pytest.warns(UserWarning, match="diff on read defaults to False"):
        ss.read(testfile, flag_choice='original', diff=False)

    ins = INS(ss)

    assert ss.flag_choice is None


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
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


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
def test_polyfit():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    ss = SS()
    ss.read(testfile, diff=True)

    ins = INS(ss, order=1)

    # Mock some data for which the polyfit is exact
    x = np.arange(1, ins.Ntimes + 1)
    for ind in range(ins.Nfreqs):
        ins.metric_array[:, ind, 0] = 3 * x + 5
    ins.metric_array.mask = np.zeros(ins.metric_array.shape, dtype=bool)
    ins.weights_array = np.ones(ins.metric_array.shape)
    ins.weights_square_array = np.copy(ins.weights_array)
    ins.metric_ms, coeffs = ins.mean_subtract(return_coeffs=True)
    test_coeffs = np.zeros((ins.order + 1, ) + ins.metric_ms.shape[1:])
    test_coeffs[0, :] = 3
    test_coeffs[1, :] = 5

    assert np.all(np.allclose(ins.metric_ms, np.zeros(ins.metric_ms.shape))), "The polyfit was not exact"
    assert np.all(np.allclose(coeffs, test_coeffs)), "The polyfit got the wrong coefficients"

    ins.metric_array[:] = np.ma.masked
    ins.metric_ms = ins.mean_subtract()
    assert np.all(ins.metric_ms.mask), "The metric_ms array was not all masked"

def test_flag_uvf_freq():
    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'
    prefix = os.path.join(DATA_PATH, '%s_test' % obs)
    flags_outfile = '%s_SSINS_flags.h5' % prefix

    ss = SS()
    ss.read(testfile, diff_freq=True)

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
        # Pretend the data is off by 1 freq
        bad_uvf.freq_array += 1
        err_uvf = ins.flag_uvf(uvf=bad_uvf)

@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
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


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read", "ignore:invalid value")
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
    ins.match_events.append((slice(0, 1), slice(1, 3), 'shape', 5))
    ins.match_events.append((slice(1, 2), slice(1, 3), 'shape', 5))
    ins.metric_array[:2, 1:3] = np.ma.masked
    ins.metric_ms = ins.mean_subtract()

    ins.write(prefix, output_type='data', clobber=True)
    ins.write(prefix, output_type='z_score', clobber=True)
    ins.write(prefix, output_type='mask', clobber=True)
    ins.write(prefix, output_type='match_events', clobber=True)
    ins.write(prefix, output_type='data', sep='.', clobber=True)
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
    #to override the fact that the data files don't have dif_ keywords set
    ins.extra_keywords['dif_time'] = True
    ins.extra_keywords['dif_freq'] = False

    mwaf_files = [os.path.join(DATA_PATH, '1061313128_12.mwaf')]
    bad_mwaf_files = [os.path.join(DATA_PATH, 'bad_file_path')]
    metafits_file = os.path.join(DATA_PATH, '1061313128.metafits')

    # Compatible shape with mwaf file
    ins.metric_array = np.ma.ones([55, 384, 1])
    ins.metric_array[50, 16 * 12: int(16 * (12 + 0.5))] = np.ma.masked

    # metadata from the input file
    NCHANS = 32
    Nbls = 8256
    NSCANS = 224

    # hard code the answer
    new_flags = np.zeros((NSCANS * Nbls, NCHANS), dtype=bool)
    new_flags[Nbls * 200:Nbls * 208, :16] = 1

    # Test some defensive errors
    with pytest.raises(IOError):
        ins.write(prefix, output_type='mwaf', mwaf_files=bad_mwaf_files,
                  metafits_file=metafits_file)
    with pytest.raises(ValueError):
        ins.write(prefix, output_type='mwaf', mwaf_files=mwaf_files,
                  mwaf_method='bad_method', metafits_file=metafits_file)
    with pytest.raises(ValueError):
        ins.write(prefix, output_type='mwaf', mwaf_files=None,
                  metafits_file=metafits_file)
    with pytest.raises(ValueError):
        ins.write(prefix, output_type='mwaf', mwaf_files=mwaf_files)

    ins.write('%s_add' % prefix, output_type='mwaf', mwaf_files=mwaf_files,
              metafits_file=metafits_file)
    ins.write('%s_replace' % prefix, output_type='mwaf', mwaf_files=mwaf_files,
              mwaf_method='replace', metafits_file=metafits_file)

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
    ins.metric_array.mask[7, :12] = True

    new_ins = ins.select(times=ins.time_array[3:-3], freq_chans=np.arange(24),
                         inplace=False)

    Ntimes = len(ins.time_array)
    ins.select(times=ins.time_array[3:-3], freq_chans=np.arange(24))

    assert ins.metric_array.shape[0] == Ntimes - 6
    assert ins.metric_array.shape[1] == 24
    for param in ins._data_params:
        assert getattr(ins, param).shape == ins.metric_array.shape
    # Check that the mask is propagated
    assert np.all(ins.metric_array.mask[4, :12])
    assert np.count_nonzero(ins.metric_array.mask) == 12

    # Check that new_ins is a copy of ins
    assert new_ins == ins


def test_data_params():
    obs = '1061313128_99bl_1pol_half_time_SSINS'
    testfile = os.path.join(DATA_PATH, '%s.h5' % obs)
    ins = INS(testfile)
    test_params = ['metric_array', 'weights_array', 'weights_square_array',
                   'metric_ms', 'sig_array']

    assert ins._data_params == test_params


def test_spectrum_type_file_init():
    obs = "1061313128_99bl_1pol_half_time_SSINS"
    auto_obs = "1061312640_mix_auto_SSINS_data"
    cross_obs = "1061312640_mix_cross_SSINS_data"
    testfile = os.path.join(DATA_PATH, f"{obs}.h5")
    auto_testfile = os.path.join(DATA_PATH, f"{auto_obs}.h5")
    cross_testfile = os.path.join(DATA_PATH, f"{cross_obs}.h5")
    ins = INS(testfile)

    assert ins.spectrum_type == "cross"

    with pytest.raises(ValueError, match="Reading in a 'cross' spectrum as 'auto'."):
        ins = INS(testfile, spectrum_type="auto")
    with pytest.raises(ValueError, match="Requested spectrum type disagrees with saved spectrum. "):
        ins = INS(auto_testfile, spectrum_type="cross")
    with pytest.raises(ValueError, match="Requested spectrum type disagrees with saved spectrum. "):
        ins = INS(cross_testfile, spectrum_type="auto")

    del ins
    ins = INS(cross_testfile)
    del ins
    ins = INS(auto_testfile, spectrum_type="auto")


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
def test_spectrum_type_bl_init():
    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, f'{obs}.uvfits')

    ss = SS()
    ss.read(testfile, diff=True)

    ins = INS(ss)
    assert "Initialized spectrum_type:cross from visibility data." in ins.history

    with pytest.raises(ValueError, match="Requested spectrum type is 'auto', but no autos exist."):
        ins = INS(ss, spectrum_type="auto")


def test_spectrum_type_bad_input():
    obs = "1061313128_99bl_1pol_half_time_SSINS"
    testfile = os.path.join(DATA_PATH, f"{obs}.h5")
    with pytest.raises(ValueError, match="Requested spectrum_type is invalid."):
        ins = INS(testfile, spectrum_type="foo")


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
def test_no_cross_auto_spectrum():
    obs = "1061312640_autos"
    testfile = os.path.join(DATA_PATH, f'{obs}.uvfits')

    ss = SS()
    ss.read(testfile, diff=True)

    with pytest.raises(ValueError, match="Requested spectrum type is 'cross', but no cross"):
        ins = INS(ss)


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
def test_mix_spectrum():
    obs = "1061312640_mix"
    testfile = os.path.join(DATA_PATH, f'{obs}.uvfits')

    ss = SS()
    ss.read(testfile, diff=True)

    with pytest.warns(UserWarning, match="Requested spectrum type is 'cross'. Removing autos before averaging."):
        ins = INS(ss)

    with pytest.warns(UserWarning, match="Requested spectrum type is 'auto'. Removing"):
        ins = INS(ss, spectrum_type="auto")

    # Hack polarization array to check error
    ss.polarization_array[0] = 1
    with pytest.raises(ValueError, match="SS input has pseudo-Stokes data. SSINS does not"):
        ins = INS(ss, spectrum_type="auto")


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read", "ignore:invalid value")
def test_use_integration_weights():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    ss = SS()
    ss.read(testfile, flag_choice='original', diff=True)

    ins = INS(ss, use_integration_weights=True)

    # These will not be equal if weights are not binary to begin with
    # The accuracy of return_weights_square is already checked in pyuvdata
    assert not np.all(ins.weights_array == ins.weights_square_array)


def test_add():
    obs = "1061313128_99bl_1pol_half_time_SSINS"
    testfile = os.path.join(DATA_PATH, f"{obs}.h5")

    truth_ins = INS(testfile)

    first_ins = INS(testfile)
    first_ins.select(freq_chans=np.arange(192))

    second_ins = INS(testfile)
    second_ins.select(freq_chans=np.arange(192, 384))

    combo_ins = first_ins.__add__(second_ins, axis='frequency')
    first_ins.__add__(second_ins, axis='frequency', inplace=True)

    # Check consistency
    assert np.all(combo_ins.metric_array.data == first_ins.metric_array.data)
    assert np.all(combo_ins.metric_array.mask == first_ins.metric_array.mask)
    assert np.all(combo_ins.metric_array.data == truth_ins.metric_array.data)
    assert np.all(combo_ins.metric_array.mask == truth_ins.metric_array.mask)

def test_mask_to_flags():
    #verify array sizes
    ss = SS()
    filepath = 'SSINS/data/1061313128_99bl_1pol_half_time.uvfits'
    ss.read(filepath, diff=True)
    ins = INS(ss)
    flags = ins.mask_to_flags()
    ss.read(filepath, diff=False, diff_freq=True)
    ins = INS(ss)
    flags = ins.mask_to_flags()
