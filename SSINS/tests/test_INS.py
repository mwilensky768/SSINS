from SSINS import INS, SS
from SSINS.data import DATA_PATH
import numpy as np
import os
import pytest
from pyuvdata import UVData, UVFlag
from datetime import datetime

@pytest.fixture
def mix_obs():
    return "1061312640_mix"

@pytest.fixture
def mix_file(mix_obs):
    return os.path.join(DATA_PATH, f"{mix_obs}.uvfits")


@pytest.fixture
def cross_obs(mix_obs):
    return f"{mix_obs}_cross_SSINS_data"


@pytest.fixture
def cross_testfile(cross_obs):
    return os.path.join(DATA_PATH, f"{cross_obs}.h5")


@pytest.fixture
def tv_obs():
    return '1061313128_99bl_1pol_half_time'


@pytest.fixture
def tv_testfile(tv_obs):
    return os.path.join(DATA_PATH, f'{tv_obs}.uvfits')

@pytest.fixture
def tv_ins_testfile(tv_obs):
    return os.path.join(DATA_PATH, f"{tv_obs}_SSINS.h5")


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:invalid value",
                            "ignore:SS.read")
def test_init(tv_testfile):

    ss = SS()
    ss.read(tv_testfile, flag_choice='original', diff=True)
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


def test_no_diff_start(tv_testfile):

    # Don't diff - will fail to mask data array
    ss = SS()
    with pytest.warns(UserWarning, match="flag_choice will be ignored"):
        ss.read(tv_testfile, flag_choice='original', diff=False)

    with pytest.warns(UserWarning, match="diff on read defaults to False"):
        ss.read(tv_testfile, flag_choice='original', diff=False)

    ins = INS(ss)

    assert ss.flag_choice is None


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
def test_mean_subtract(tv_testfile):

    ss = SS()
    ss.read(tv_testfile, diff=True)

    ins = INS(ss, order=0)

    old_dat = np.copy(ins.metric_ms)

    # Mask the first five frequencies and last two at the first and second times
    ins.metric_array[0, :5] = np.ma.masked

    # Calculate the new mean-subtracted spectrum only over the first few masked frequencies
    ins.metric_ms[:, :5] = ins.mean_subtract(freq_slice=slice(0, 5))

    # See if a new mean was calculated over the first five frequencies
    assert not np.all(old_dat[1:, :5] == ins.metric_ms[1:, :5]), "All elements of the ms array are still equal"


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
def test_polyfit(tv_testfile):

    ss = SS()
    ss.read(tv_testfile, diff=True)

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


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
def test_mask_to_flags(tmp_path, tv_obs, tv_testfile):

    prefix = os.path.join(tmp_path, f'{tv_obs}_test')
    flags_outfile = f'{prefix}_SSINS_flags.h5'

    ss = SS()
    ss.read(tv_testfile, diff=True)

    uvd = UVData()
    uvd.read(tv_testfile)

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
    assert read_uvf == uvf, "UVFlag object differs after read"


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read", "ignore:invalid value")
def test_write(tmp_path, tv_obs, tv_testfile):


    prefix = os.path.join(tmp_path, f'{tv_obs}_test')
    data_outfile = f'{prefix}_SSINS_data.h5'
    z_score_outfile = f'{prefix}_SSINS_z_score.h5'
    mask_outfile = f'{prefix}_SSINS_mask.h5'
    match_outfile = f'{prefix}_SSINS_match_events.yml'
    sep_data_outfile = f'{prefix}.SSINS.data.h5'

    ss = SS()
    ss.read(tv_testfile, flag_choice='original', diff=True)

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
    assert os.path.exists(sep_data_outfile), "sep_data_outfile was not written"
    assert os.path.exists(z_score_outfile)


def test_write_mwaf(tmp_path, tv_obs, tv_ins_testfile):
    from astropy.io import fits
 
    prefix = os.path.join(tmp_path, f'{tv_obs}_SSINS_test')
    ins = INS(tv_ins_testfile)
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

    ins.write(f'{prefix}_add', output_type='mwaf', mwaf_files=mwaf_files,
              metafits_file=metafits_file)
    ins.write(f'{prefix}_replace', output_type='mwaf', mwaf_files=mwaf_files,
              mwaf_method='replace', metafits_file=metafits_file)

    with fits.open(mwaf_files[0]) as old_mwaf_hdu:
        with fits.open(f'{prefix}_add_12.mwaf') as add_mwaf_hdu:
            assert np.all(add_mwaf_hdu[1].data['FLAGS'] == old_mwaf_hdu[1].data['FLAGS'] + new_flags)
    with fits.open(f'{prefix}_replace_12.mwaf') as replace_mwaf_hdu:
        assert np.all(replace_mwaf_hdu[1].data['FLAGS'] == new_flags)


def test_select(tv_ins_testfile):

    ins = INS(tv_ins_testfile)
    ins.metric_array.mask[7, :12] = True

    with pytest.warns(UserWarning, match="sig_array has been reset"):
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


def test_data_params(tv_ins_testfile):

    ins = INS(tv_ins_testfile)
    test_params = ['metric_array', 'weights_array', 'weights_square_array',
                   'metric_ms', 'sig_array']

    assert ins._data_params == test_params


def test_spectrum_type_file_init(cross_testfile, tv_ins_testfile):

    auto_obs = "1061312640_mix_auto_SSINS_data"
    auto_testfile = os.path.join(DATA_PATH, f"{auto_obs}.h5")

    ins = INS(tv_ins_testfile)

    assert ins.spectrum_type == "cross"

    with pytest.raises(ValueError, match="Requested spectrum type disagrees with saved spectrum. "):
        ins = INS(auto_testfile, spectrum_type="cross")
    with pytest.raises(ValueError, match="Requested spectrum type disagrees with saved spectrum. "):
        ins = INS(cross_testfile, spectrum_type="auto")

    del ins
    ins = INS(cross_testfile) # I think this line just gets coverage?
    del ins
    ins = INS(auto_testfile, spectrum_type="auto")

@pytest.mark.filterwarnings("ignore:channel_width", "ignore:telescope_name", "ignore:Antenna", "ignore:telescope_location")
def test_old_file():
    
    old_ins_file = os.path.join(DATA_PATH, "1090867840_SSINS_data.h5")
    old_mask_file = os.path.join(DATA_PATH, "1090867840_SSINS_mask.h5")
    
    try:
        # this works with pyuvdata>=3.0
        with pytest.raises(
            ValueError, match="Required UVParameter _Nants has not been set."
        ):
            ins = INS(old_ins_file, mask_file=old_mask_file)
    
    except AssertionError:
        # this works with pyuvdata<3.0
        with pytest.raises(
            ValueError, match="Required UVParameter _antenna_names has not been set."
        ):
            ins = INS(old_ins_file, mask_file=old_mask_file)


    with pytest.raises(ValueError, 
                       match="spectrum_type is set to auto, but file input is a cross spectrum from an old file."):
        ins = INS(old_ins_file, telescope_name="mwa", spectrum_type="auto")
    
    # Just check that it reads
    ins = INS(old_ins_file, telescope_name="mwa", mask_file=old_mask_file)
    assert ins is not None
    


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
def test_spectrum_type_bl_init(tv_testfile):

    ss = SS()
    ss.read(tv_testfile, diff=True)

    ins = INS(ss)
    assert "Initialized spectrum_type:cross from visibility data." in ins.history

    with pytest.raises(ValueError, match="Requested spectrum type is 'auto', but no autos exist."):
        ins = INS(ss, spectrum_type="auto")


def test_spectrum_type_bad_input(tv_ins_testfile):

    with pytest.raises(ValueError, match="Requested spectrum_type is invalid."):
        ins = INS(tv_ins_testfile, spectrum_type="foo")


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
def test_no_cross_auto_spectrum():
    obs = "1061312640_autos"
    testfile = os.path.join(DATA_PATH, f'{obs}.uvfits')

    ss = SS()
    ss.read(testfile, diff=True)

    with pytest.raises(ValueError, match="Requested spectrum type is 'cross', but no cross"):
        ins = INS(ss)


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
def test_mix_spectrum(mix_file):

    ss = SS()
    ss.read(mix_file, diff=True)

    with pytest.warns(UserWarning, match="Requested spectrum type is 'cross'. Removing autos before averaging."):
        ins = INS(ss)

    with pytest.warns(UserWarning, match="Requested spectrum type is 'auto'. Removing"):
        ins = INS(ss, spectrum_type="auto")

    # Hack polarization array to check error
    ss.polarization_array[0] = 1
    with pytest.raises(ValueError, match="SS input has pseudo-Stokes data. SSINS does not"):
        ins = INS(ss, spectrum_type="auto")


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read", "ignore:invalid value")
def test_use_integration_weights(tv_testfile):

    ss = SS()
    ss.read(tv_testfile, flag_choice='original', diff=True)

    ins = INS(ss, use_integration_weights=True)

    # These will not be equal if weights are not binary to begin with
    # The accuracy of return_weights_square is already checked in pyuvdata
    assert not np.all(ins.weights_array == ins.weights_square_array)


def test_add(tv_ins_testfile):

    truth_ins = INS(tv_ins_testfile)

    first_ins = INS(tv_ins_testfile)
    first_ins.select(freq_chans=np.arange(192))

    second_ins = INS(tv_ins_testfile)
    second_ins.select(freq_chans=np.arange(192, 384))

    combo_ins = first_ins.__add__(second_ins, axis='frequency')
    first_ins.__add__(second_ins, axis='frequency', inplace=True)

    # Check consistency
    assert np.all(combo_ins.metric_array.data == first_ins.metric_array.data)
    assert np.all(combo_ins.metric_array.mask == first_ins.metric_array.mask)
    assert np.all(combo_ins.metric_array.data == truth_ins.metric_array.data)
    assert np.all(combo_ins.metric_array.mask == truth_ins.metric_array.mask)

def test_read_from_instance(cross_testfile):
    ins = INS(cross_testfile)

    with pytest.raises(NotImplementedError, match="SSINS does not currently support "):
        ins.read(cross_testfile)


def test_set_weights_square_array(cross_testfile):

    ins = INS(cross_testfile)
    copy_ins = ins.copy()
    
    ins.weights_square_array = None
    ins.set_ins_data_params()

    assert ins.weights_square_array is not None
    assert np.array_equal(ins.metric_ms, copy_ins.metric_ms) # check that the goods are intact