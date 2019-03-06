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

    # Make up a simple event list with five shapes, where shape_a is redundant
    # with shape_b and shape_c with shape_d
    match_events = [(0, 0, slice(10, 20), 'shape_a'), (0, 0, slice(9, 21), 'shape_b'),
                    (1, 0, slice(9, 21), 'shape_b'), (2, 0, slice(10, 20), 'shape_a'),
                    (0, 0, slice(30, 40), 'shape_c'), (2, 0, slice(20, 30), 'shape_e'),
                    (0, 0, slice(29, 31), 'shape_d')]

    # The first and last event should be removed if we prioritize shape_b over shape_a and shape_c over shape_d
    answer = [(0, 0, slice(9, 21), 'shape_b'), (1, 0, slice(9, 21), 'shape_b'),
              (2, 0, slice(10, 20), 'shape_a'), (0, 0, slice(30, 40), 'shape_c'),
              (2, 0, slice(20, 30), 'shape_e')]

    test_output = util.red_event_sort(match_events, [('shape_a', 'shape_b'), ('shape_d', 'shape_c')], keep_prior=[1, 0])
    nt.eq_(test_output, answer)


def test_match_fraction():
    # Make up a simple event list belonging to some fictitious data with 5 times and 100 frequencies
    events = np.array([(1, 0, slice(0, 10)), (2, 0, slice(0, 10)), (3, 0, slice(10, 20))])
    Ntimes = 5
    Nfreqs = 100
    # Make the event_fraction dictionary
    event_frac = util.event_fraction(events, Nfreqs, Ntimes)
    nt.ok_(event_frac == {(0, 10): 2 / 5, (10, 20): 1 / 5})
