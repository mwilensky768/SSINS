import pytest
from SSINS.data import DATA_PATH
from SSINS import SS, INS
import os
import numpy as np
from pyuvdata import UVData

"""
Tests the various capabilities of the sky_subtract class
"""


def test_SS_read():
    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    ss = SS()

    # Test reading in only metadata skips if block
    ss.read(testfile, read_data=False)
    assert ss.data_array is None, "Data array is not None"

    # Test select on read and diff
    ss.read(testfile, times=np.unique(ss.time_array)[1:10], diff=True)
    assert ss.Ntimes == 8, "Diff seems like it wasn't executed correctly"

    # See that it still passes UVData check
    assert ss.check()


def test_apply_flags():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'
    insfile = os.path.join(DATA_PATH, '%s_SSINS.h5' % obs)
    ss = SS()

    ss.read(testfile, diff=True)

    # Make sure no flags are applied to start with
    assert not np.any(ss.data_array.mask), "There are some flags to start with."

    # Apply flags, test equality, test attribute change
    ss.apply_flags(flag_choice='original')
    assert np.all(ss.flag_array == ss.data_array.mask), "Flag arrays are not equal"
    assert ss.flag_choice is 'original', "Flag choice attribute was not changed"

    # Revert flags back, test equality, test attribute change
    ss.apply_flags(flag_choice=None)
    assert not np.any(ss.data_array.mask), "Flags did not revert back back"
    assert ss.flag_choice is None, "Flag choice attribute did not revert back"

    # Make a custom flag array where everything is flagged, check application
    custom = np.ones_like(ss.flag_array)
    ss.apply_flags(flag_choice='custom', custom=custom)
    assert np.all(ss.data_array.mask), "The custom flag array was not applied"
    assert ss.flag_choice is 'custom', "The flag choice attribute was not changed"

    # Read an INS in (no flags by default) and flag a channel, test if it applies correctly
    ins = INS(insfile)
    ins.metric_array.mask[:, 0] = True
    ss.apply_flags(flag_choice='INS', INS=ins)
    assert np.all(ss.data_array.mask[:, 0, 0]), "Not all of the 0th channel was flagged."
    assert not np.any(ss.data_array.mask[:, 0, 1:]), "Some of the channels other than the 0th were flagged"
    assert ss.flag_choice is 'INS'

    # Make flag_choice custom but do not provide array - should unflag everything and issue a warning
    with pytest.warns(UserWarning, match="Custom flags were chosen, but custom flags were None type. Setting flag_choice to None and unmasking data."):
        ss.apply_flags(flag_choice='custom', custom=None)
    assert not np.any(ss.data_array.mask), "Some of the channels were still flagged"
    assert ss.flag_choice is None

    with pytest.raises(ValueError):
        ss.apply_flags(flag_choice='bad_choice')


def test_mixture_prob():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    ss = SS()
    ss.read(testfile, diff=True)
    ss.apply_flags('original')

    # Generate the mixture probabilities
    mixture_prob = ss.mixture_prob(bins='auto')

    # Check that they sum to close to 1
    assert np.isclose(np.sum(mixture_prob), 1), "Probabilities did not add up to close to 1"


def test_rev_ind():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'

    ss = SS()
    ss.read(testfile, diff=True)

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


def test_write():

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, '%s.uvfits' % obs)
    file_type = 'uvfits'
    outfile = os.path.join(DATA_PATH, 'test_write.uvfits')

    ss = SS()
    ss.read(testfile, diff=True)

    custom = np.zeros_like(ss.data_array.mask)
    custom[:ss.Nbls] = 1

    # Flags the first time and no others
    ss.apply_flags(flag_choice='custom', custom=custom)

    # Write this out without combining flags, will issue a warning
    with pytest.warns(UserWarning, match="Some nsamples are 0, which will result in failure to propagate flags. Setting nsample to default values where 0."):
        ss.write(outfile, 'uvfits', filename_in=testfile, combine=False)

    # Check if the flags propagated correctly
    UV = UVData()
    UV.read(outfile)
    assert np.all(UV.flag_array[:2 * UV.Nbls]), "Not all expected flags were propagated"
    assert not np.any(UV.flag_array[2 * UV.Nbls:]), "More flags were made than expected"
    os.remove(outfile)


def test_read_multifiles():
    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, f'{obs}.uvfits')
    new_fp1 = os.path.join(DATA_PATH, f'{obs}_new1.uvfits')
    new_fp2 = os.path.join(DATA_PATH, f'{obs}_new2.uvfits')
    flist = [new_fp1, new_fp2]

    file_type = 'uvfits'

    # Read in a file's metadata and split it into two objects
    uvd_full = UVData()
    uvd_full.read(testfile, read_data=False)
    times1 = np.unique(uvd_full.time_array)[:14]
    times2 = np.unique(uvd_full.time_array)[14:]

    # Write two separate files to be read in later
    uvd_split1 = UVData()
    uvd_split2 = UVData()
    uvd_split1.read(testfile, times=times1)
    uvd_split2.read(testfile, times=times2)
    uvd_split1.write_uvfits(new_fp1)
    uvd_split2.write_uvfits(new_fp2)

    # Check wanings and diff's
    ss_orig = SS()
    ss_multi = SS()
    # test warning raise
    with pytest.warns(UserWarning, match=("diff on read defaults to False now. Please double"
                                          " check SS.read call and ensure the appropriate"
                                          " keyword arguments for your intended use case.")):
        ss_orig.read(testfile, diff=False)
        ss_orig.diff()
    with pytest.warns(UserWarning, match=("diff on read defaults to False now. Please double"
                                          " check SS.read call and ensure the appropriate"
                                          " keyword arguments for your intended use case.")):
        ss_multi.read(flist, diff=True)

    assert np.all(np.isclose(ss_orig.data_array, ss_multi.data_array)), "Diffs were different!"
    assert ss_multi.Ntimes == (uvd_full.Ntimes - 1), "Too many diffs were done"

    for path in flist:
        os.remove(path)
