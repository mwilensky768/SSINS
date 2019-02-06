from __future__ import division, print_function, absolute_import

from SSINS import util
from SSINS.data import DATA_PATH
import nose.tools as nt
import os
import numpy as np
import scipy.stats


def test_obslist():
    obsfile = os.path.join(DATA_PATH, 'obs_list.txt')
    obslist_test = ['1061313008', '1061313128', '1061318864', '1061318984']
    obslist = util.make_obslist(obsfile)
    nt.eq_(obslist_test, obslist)


def test_red_event_sort():

    # Make up a simple event list with three shapes, where shape_a and shape_b are redundant
    match_events = [(0, 0, slice(10, 20), 'shape_a'), (0, 0, slice(9, 21), 'shape_b'),
                    (1, 0, slice(9, 21), 'shape_b'), (2, 0, slice(10, 20), 'shape_a'),
                    (0, 0, slice(30, 40), 'shape_c')]

    # The first event should be removed if we prioritize shape_b over shape_a
    answer = [(0, 0, slice(9, 21), 'shape_b'), (1, 0, slice(9, 21), 'shape_b'),
              (2, 0, slice(10, 20), 'shape_a'), (0, 0, slice(30, 40), 'shape_c')]

    test_output = util.red_event_sort(match_events, [('shape_a', 'shape_b')], keep_prior=[1, 0])
    nt.eq_(test_output, answer)


def test_match_fraction():
    # Make up a simple event list belonging to some fictitious data with 5 times and 100 frequencies
    events = np.array([(1, 0, slice(0, 10)), (2, 0, slice(0, 10)), (3, 0, slice(10, 20))])
    Ntimes = 5
    Nfreqs = 100
    # Make the event_fraction dictionary
    event_frac = util.event_fraction(events, Nfreqs, Ntimes)
    nt.ok_(event_frac == {(0, 10): 2 / 5, (10, 20): 1 / 5})


def test_chisq():
    # Use bins that are typical in match_filter case
    bins = np.arange(-4, 5)
    # Make up some counts
    counts = np.array([1, 2, 5, 10, 10, 5, 2, 1])
    # Check default settings
    stat, p = util.chisq(counts, bins)
    # These happen to be the answers
    nt.ok_(np.allclose((stat, p), (3.476106234440926, 0.06226107945215504)))
    # Check expected counts weighting
    stat, p = util.chisq(counts, bins, weight='exp', thresh=5)
    # These happen to be the answers
    nt.ok_(np.allclose((stat, p), (2.6882672697527807, 0.1010896885610924)))
