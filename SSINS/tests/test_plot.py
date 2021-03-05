from SSINS import Catalog_Plot as cp
from SSINS import SS, INS, MF
from SSINS.data import DATA_PATH
import numpy as np
import os
import pytest


@pytest.mark.filterwarnings("ignore:default base")
def test_INS_plot():

    matplotlib = pytest.importorskip("matplotlib")

    obs = '1061313128_99bl_1pol_half_time'
    insfile = os.path.join(DATA_PATH, f'{obs}_SSINS.h5')
    outdir = os.path.join(DATA_PATH, 'test_plots')

    prefix = f'{outdir}/{obs}_raw'
    outfile = f'{prefix}_SSINS.pdf'
    log_prefix = f'{outdir}/{obs}_log'
    log_outfile = f'{log_prefix}_SSINS.pdf'
    symlog_prefix = f'{outdir}/{obs}_symlog'
    symlog_outfile = f'{symlog_prefix}_SSINS.pdf'

    ins = INS(insfile)

    xticks = np.arange(0, 384, 96)
    xticklabels = ['%.1f' % (10**-6 * ins.freq_array[tick]) for tick in xticks]
    yticks = np.arange(0, 50, 10)
    yticklabels = ['%i' % (2 * tick) for tick in yticks]

    # Make lst straddle the 2pi boundary to trigger the warning
    mean_lst = np.mean(ins.lst_array)
    ins.lst_array += (2 * np.pi) - mean_lst
    ins.lst_array[ins.lst_array > (2 * np.pi)] -= 2 * np.pi
    with pytest.warns(UserWarning, match="LSTs appear to cross"):
        cp.INS_plot(ins, prefix, backend='Agg', use_extent=True,
                    extent_time_format='lst')
    with pytest.warns(UserWarning, match="Plotting keyword"):
        cp.INS_plot(ins, log_prefix, log=True, xticks=xticks, yticks=yticks,
                    xticklabels=xticklabels, yticklabels=yticklabels,
                    title='Title')
    cp.INS_plot(ins, symlog_prefix, symlog=True, xticks=xticks, yticks=yticks,
                xticklabels=xticklabels, yticklabels=yticklabels,
                use_extent=False)

    assert os.path.exists(outfile), "The first plot was not made"
    assert os.path.exists(log_outfile), "The second plot was not made"
    assert os.path.exists(symlog_outfile), "The third plot was not made"

    os.remove(outfile)
    os.remove(log_outfile)
    os.remove(symlog_outfile)
    os.rmdir(outdir)


def test_sig_plot():

    matplotlib = pytest.importorskip("matplotlib")

    obs = '1061313128_99bl_1pol_half_time'
    insfile = os.path.join(DATA_PATH, f'{obs}_SSINS.h5')
    outdir = os.path.join(DATA_PATH, 'test_plots')

    prefix = '%s/%s_flagged' % (outdir, obs)
    dataplotfile = '%s_SSINS.pdf' % prefix
    outfile = '%s_SSINS_sig.pdf' % prefix

    ins = INS(insfile)
    shape_dict = {'TV6': [1.74e8, 1.81e8],
                  'TV7': [1.81e8, 1.88e8],
                  'TV8': [1.88e8, 1.95e8]}
    sig_thresh = {'TV6': 5, 'TV7': 5, 'TV8': 5, 'narrow': 5, 'streak': 5}
    mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict)
    mf.apply_match_test(ins)

    xticks = np.arange(0, 384, 96)
    xticklabels = ['%.1f' % (10**-6 * ins.freq_array[tick]) for tick in xticks]
    yticks = np.arange(0, 50, 10)
    yticklabels = ['%i' % (2 * tick) for tick in yticks]

    cp.INS_plot(ins, prefix)

    assert os.path.exists(outfile), "The first plot was not made"
    assert os.path.exists(dataplotfile), "The second plot was not made"

    os.remove(outfile)
    os.remove(dataplotfile)
    os.rmdir(outdir)


@pytest.mark.filterwarnings("ignore:SS.read", "ignore:Reordering")
def test_VDH_plot():

    matplotlib = pytest.importorskip("matplotlib")

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, f'{obs}.uvfits')
    outdir = os.path.join(DATA_PATH, 'test_plots')

    prefix = f'{outdir}/{obs}'
    outfile = f'{prefix}_VDH.pdf'

    dens_prefix = f'{outdir}/{obs}_dens'
    dens_outfile = f'{dens_prefix}_VDH.pdf'

    ss = SS()
    ss.read(testfile, flag_choice='original', diff=True)

    cp.VDH_plot(ss, prefix, backend='Agg')
    # Test with density prefix and error bars
    cp.VDH_plot(ss, dens_prefix, density=True, error_sig=1, ylim=[1e-5, 1e5], pre_model_label='model label')

    assert os.path.exists(outfile), "The first plot was not made"
    assert os.path.exists(dens_outfile), "The second plot was not made"

    os.remove(outfile)
    os.remove(dens_outfile)
    os.rmdir(outdir)


@pytest.mark.filterwarnings("ignore:SS.read", "ignore:Reordering")
def test_VDH_no_model():

    matplotlib = pytest.importorskip("matplotlib")

    obs = '1061313128_99bl_1pol_half_time'
    testfile = os.path.join(DATA_PATH, f'{obs}.uvfits')
    outdir = os.path.join(DATA_PATH, 'test_plots')
    prefix = f'{outdir}/{obs}'
    outfile = f'{prefix}_VDH.pdf'

    ss = SS()
    ss.read(testfile, flag_choice=None, diff=True)

    with pytest.warns(UserWarning, match="Asking to plot post-flagging data, but SS.flag_choice is None. This is identical to plotting pre-flagging data"):
        cp.VDH_plot(ss, prefix, pre_model=False, post_model=False)

    assert os.path.exists(outfile), "The plot was not made"

    os.remove(outfile)
    os.rmdir(outdir)
