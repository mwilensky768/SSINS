import pytest
from SSINS.data import DATA_PATH
from SSINS import SS, INS
import os
import numpy as np
from pyuvdata import UVData

"""
Tests the various capabilities of the sky_subtract class
"""

@pytest.fixture
def tv_obs():
    return '1061313128_99bl_1pol_half_time'


@pytest.fixture
def tv_testfile(tv_obs):
    return os.path.join(DATA_PATH, f'{tv_obs}.uvfits')


@pytest.mark.filterwarnings("ignore:Reordering", "ignore:SS.read")
def test_SS_read(tv_testfile):

    ss = SS()

    # Test reading in only metadata skips if block and warning
    with pytest.warns(PendingDeprecationWarning, match="SS.read will be renamed"):
        ss.read(tv_testfile, read_data=False)
    assert ss.data_array is None, "Data array is not None"

    # Test select on read and diff
    ss.read(tv_testfile, times=np.unique(ss.time_array)[1:10], diff=True)
    assert ss.Ntimes == 8, "Number of times after diff disagrees!"
    assert ss.Nbls == 99, "Number of baselines is incorrect"

    # See that it still passes UVData check
    assert ss.check()


def test_diff(tv_testfile):

    ss = SS()
    uv = UVData()

    # Read in two times and two baselines of data, so that the diff is obvious.
    uv.read(tv_testfile, read_data=False)
    times = np.unique(uv.time_array)[:2]
    bls = [(uv.antenna_numbers[0], uv.antenna_numbers[1]),
           (uv.antenna_numbers[0], uv.antenna_numbers[2])]
    uv.read(tv_testfile, times=times, bls=bls)
    uv.reorder_blts(order='baseline')

    diff_dat = uv.data_array[1::2] - uv.data_array[::2]
    diff_flags = np.logical_or(uv.flag_array[::2], uv.flag_array[1::2])
    diff_times = 0.5 * (uv.time_array[::2] + uv.time_array[1::2])
    diff_nsamples = 0.5 * (uv.nsample_array[::2] + uv.nsample_array[1::2])
    diff_ints = uv.integration_time[::2] + uv.integration_time[1::2]
    diff_uvw = 0.5 * (uv.uvw_array[::2] + uv.uvw_array[1::2])
    diff_pcad = 0.5 * (uv.phase_center_app_dec[::2] + uv.phase_center_app_dec[1::2])
    diff_pcar = 0.5 * (uv.phase_center_app_ra[::2] + uv.phase_center_app_ra[1::2])
    diff_pcfp = 0.5 * (uv.phase_center_frame_pa[::2] + uv.phase_center_frame_pa[1::2])

    with pytest.warns(UserWarning, match="Reordering data array to baseline order to perform differencing."):
        ss.read(tv_testfile, diff=True, times=times, bls=bls)
    ss.reorder_blts(order='baseline')

    assert np.all(ss.data_array == diff_dat), "Data values are different!"
    assert np.all(ss.flag_array == diff_flags), "Flags are different!"
    assert np.all(ss.time_array == diff_times), "Times are different!"
    assert np.all(ss.nsample_array == diff_nsamples), "nsample_array is different!"
    assert np.all(ss.integration_time == diff_ints), "Integration times are different"
    assert np.all(ss.uvw_array == diff_uvw), "uvw_arrays disagree!"
    assert np.all(ss.ant_1_array == np.array([uv.antenna_numbers[0], uv.antenna_numbers[0]])), f"ant_1_array disagrees!"
    assert np.all(ss.ant_2_array == np.array([uv.antenna_numbers[1], uv.antenna_numbers[2]])), "ant_2_array disagrees!"
    assert np.all(ss.phase_center_app_dec == diff_pcad)
    assert np.all(ss.phase_center_app_ra == diff_pcar)
    assert np.all(ss.phase_center_frame_pa == diff_pcfp)


@pytest.mark.filterwarnings("ignore:SS.read", "ignore:Reordering")
def test_apply_flags(tv_testfile):

    insfile = os.path.join(DATA_PATH, '%s_SSINS.h5' % obs)
    ss = SS()

    ss.read(tv_testfile, diff=True)

    # Make sure no flags are applied to start with
    assert not np.any(ss.data_array.mask), "There are some flags to start with."

    # Apply flags, test equality, test attribute change
    ss.apply_flags(flag_choice='original')
    assert np.all(ss.flag_array == ss.data_array.mask), "Flag arrays are not equal"
    assert ss.flag_choice == 'original', "Flag choice attribute was not changed"

    # Revert flags back, test equality, test attribute change
    ss.apply_flags(flag_choice=None)
    assert not np.any(ss.data_array.mask), "Flags did not revert back back"
    assert ss.flag_choice is None, "Flag choice attribute did not revert back"

    # Make a custom flag array where everything is flagged, check application
    custom = np.ones_like(ss.flag_array)
    ss.apply_flags(flag_choice='custom', custom=custom)
    assert np.all(ss.data_array.mask), "The custom flag array was not applied"
    assert ss.flag_choice == 'custom', "The flag choice attribute was not changed"

    # Read an INS in (no flags by default) and flag a channel for two times stuff, see if applied correctly
    ins = INS(insfile)
    ins.metric_array.mask[[2, 4], 1, :] = True
    ss.apply_flags(flag_choice='INS', INS=ins)
    assert np.all(ss.data_array.mask[2::ss.Ntimes, :, 1, :]), "The 2nd time was not flagged."
    assert np.all(ss.data_array.mask[4::ss.Ntimes, :, 1, :]), "The 4th time was not flagged."
    assert not np.any(ss.data_array.mask[:, :, [0] + list(range(2, ss.Nfreqs)), :]), "Channels were flagged that should not have been."
    assert ss.flag_choice == 'INS'

    # Make a bad time array to test an error
    ins.time_array = ins.time_array + 1
    with pytest.raises(ValueError):
        ss.apply_flags(flag_choice='INS', INS=ins)

    # Make flag_choice custom but do not provide array - should unflag everything and issue a warning
    with pytest.warns(UserWarning, match="Custom flags were chosen, but custom flags were None type. Setting flag_choice to None and unmasking data."):
        ss.apply_flags(flag_choice='custom', custom=None)
    assert not np.any(ss.data_array.mask), "Some of the channels were still flagged"
    assert ss.flag_choice is None

    with pytest.raises(ValueError):
        ss.apply_flags(flag_choice='bad_choice')

@pytest.mark.filterwarnings("ignore:SS.read", "ignore:Reordering")
def test_apply_flags_on_diff(tv_testfile):
    ss = SS()

    ss.read(tv_testfile, diff=True, flag_choice='original')

    assert np.all(ss.flag_array == ss.data_array.mask), "Flag arrays are not equal"
    assert ss.flag_choice == 'original', "Flag choice attribute was not changed"


@pytest.mark.filterwarnings("ignore:SS.read", "ignore:Reordering",
                            "ignore:diff on read")
def test_mixture_prob(tv_testfile):

    ss = SS()
    ss.read(tv_testfile, diff=True)
    ss.apply_flags('original')

    # Generate the mixture probabilities
    mixture_prob = ss.mixture_prob(bins='auto')

    # Check that they sum to close to 1
    assert np.isclose(np.sum(mixture_prob), 1), "Probabilities did not add up to close to 1"

    # Do a new read, but don't diff. Run and check mask.
    ss = SS()
    ss.read(tv_testfile, diff=False)

    mixture_prob = ss.mixture_prob(bins='auto')

    assert ss.flag_choice is None


@pytest.mark.filterwarnings("ignore:SS.read", "ignore:Reordering",
                            "ignore:diff on read")
def test_rev_ind():

    ss = SS()
    ss.read(tv_testfile, diff=True)

    # Make a band that will pick out only the largest value in the data
    dat_sort = np.sort(np.abs(ss.data_array), axis=None)
    band = [0.5 * (dat_sort[-2] + dat_sort[-1]), dat_sort[-1] + 1]

    # Find the indices of this data point
    ind = np.unravel_index(np.absolute(ss.data_array).argmax(), ss.data_array.shape)
    # Convert the blt to a time index
    t = ind[0] // ss.Nbls
    f = ind[2]
    p = ind[3]

    # Make the waterfall histogram
    wf_hist = ss.rev_ind(band)

    # Check that it picked up that point
    assert wf_hist[t, f, p] == 1, "The algorithm did not find the data point"

    # Check no other points were picked up
    assert np.count_nonzero(wf_hist) == 1, "The algorithm found other data"

    # Do a new read, but don't diff. Run and check mask.
    ss = SS()
    ss.read(tv_testfile, diff=False)

    rev_ind_hist = ss.rev_ind(band)

    assert ss.flag_choice is None


@pytest.mark.filterwarnings("ignore:SS.read", "ignore:Reordering",
                            "ignore:some nsamples", "ignore:elementwise")
def test_write(tmp_path, tv_testfile):

    outfile = os.path.join(tmp_path, 'test_write.uvfits')

    ss = SS()
    ss.read(tv_testfile, diff=True)

    blt_inds = np.where(ss.time_array == np.unique(ss.time_array)[10])
    custom = np.zeros_like(ss.data_array.mask)
    custom[blt_inds, :, 64:128, :] = 1

    # Flags the first time and no others
    ss.apply_flags(flag_choice='custom', custom=custom)

    # Write this out without combining flags, will issue a warning
    with pytest.warns(UserWarning, match="Some nsamples are 0, which will result in failure to propagate flags. Setting nsample to default values where 0."):
        ss.write(outfile, 'uvfits', filename_in=tv_testfile, combine=False)

    # Check if the flags propagated correctly
    UV = UVData()
    UV.read(outfile)
    blt_inds = np.isin(UV.time_array, np.unique(UV.time_array)[10:12])
    assert np.all(UV.flag_array[blt_inds, :, 64:128, :]), "Not all expected flags were propagated"

    new_blt_inds = np.logical_not(np.isin(UV.time_array, np.unique(UV.time_array)[10:12]))
    assert not np.any(UV.flag_array[new_blt_inds, :, 64:128, :]), "More flags were made than expected"

    # Test bad read.
    bad_uv_filepath = os.path.join(DATA_PATH, '1061312640_mix.uvfits')
    bad_uv = UVData()
    bad_uv.read(bad_uv_filepath)
    with pytest.raises(ValueError, match="UVData and SS objects were found to be incompatible."):
        ss.write(outfile, 'uvfits', bad_uv)


def test_read_multifiles(tmp_path, tv_obs, tv_testfile):

    new_fp1 = os.path.join(tmp_path, f'{tv_obs}_new1.uvfits')
    new_fp2 = os.path.join(tmp_path, f'{tv_obs}_new2.uvfits')
    flist = [new_fp1, new_fp2]

    # Read in a file's metadata and split it into two objects
    uvd_full = UVData()
    uvd_full.read(tv_testfile, read_data=False)
    times1 = np.unique(uvd_full.time_array)[:14]
    times2 = np.unique(uvd_full.time_array)[14:]

    # Write two separate files to be read in later
    uvd_split1 = UVData()
    uvd_split2 = UVData()
    uvd_split1.read(tv_testfile, times=times1)
    uvd_split2.read(tv_testfile, times=times2)
    uvd_split1.write_uvfits(new_fp1)
    uvd_split2.write_uvfits(new_fp2)

    # Check wanings and diff's
    ss_orig = SS()
    ss_multi = SS()
    # test warning raise
    with pytest.warns(UserWarning, match=("diff on read defaults to False now. Please double"
                                          " check SS.read call and ensure the appropriate"
                                          " keyword arguments for your intended use case.")):
        ss_orig.read(tv_testfile, diff=False)
        ss_orig.diff()
        ss_multi.read(flist, diff=True)

    assert np.all(np.isclose(ss_orig.data_array, ss_multi.data_array)), "Diffs were different!"
    assert ss_multi.Ntimes == (uvd_full.Ntimes - 1), "Too many diffs were done"


@pytest.mark.filterwarnings("ignore:SS.read", "ignore:diff on read")
def test_newmask(tv_testfile):

    ss = SS()
    ss.read(tv_testfile, diff=False)

    assert not isinstance(ss.data_array, np.ma.MaskedArray)

    ss.apply_flags()

    assert ss.flag_choice is None
    assert isinstance(ss.data_array, np.ma.MaskedArray)


def test_Nphase_gt_1(tmp_path, tv_testfile):
    uvd = UVData()
    uvd.read(tv_testfile, read_data=False)

    # Split the object so we can phase to separate locations
    unique_times = np.unique(uvd.time_array)
    first_times = unique_times[:10]
    last_times = unique_times[-10:]

    uvfirst = UVData()
    uvfirst.read(tv_testfile, times=first_times)

    uvlast = UVData()
    uvlast.read(tv_testfile, times=last_times)

    # Adjust phase of one object and write new file
    og_pc_ra = uvd.phase_center_app_ra[0]
    og_pc_dec = uvd.phase_center_app_dec[0]

    phase_shift = 1e-2
    uvfirst.phase(ra=og_pc_ra + phase_shift, dec=og_pc_dec + phase_shift)

    uv_nphase2 = uvfirst + uvlast

    assert uv_nphase2.Nphase == 2

    nphase2_file = os.path.join(tmp_dir, "test_nphase2.uvh5")
    uv_nphase2.write_uvh5(nphase2_file)

    ss = SS()
    with pytest.raises(NotImplementedError, match="SSINS cannot handle files with more than one phase center."):
        ss.read(nphase2_file)



