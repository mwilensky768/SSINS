from __future__ import division, print_function, absolute_import

from SSINS import util
from SSINS.data import DATA_PATH
import nose.tools as nt
import os
import numpy as np
import scipy.stats


def test_obslist():
    obsfile = os.path.join(DATA_PATH, 'obs_list.txt')
    outfile = os.path.join(DATA_PATH, 'test_obs_list.txt')
    obslist_test = ['1061313008', '1061313128', '1061318864', '1061318984']
    obslist = util.make_obslist(obsfile)
    util.make_obsfile(obslist, outfile)
    obslist_test_2 = util.make_obslist(outfile)

    assert obslist_test == obslist, "The lists were not equal"
    assert os.path.exists(outfile), "A file was not written"
    assert obslist_test == obslist_test_2, "The new file did not read in properly"

    os.remove(outfile)


def test_red_event_sort():

    # Make up a simple event list with five shapes, where shape_a is redundant
    # with shape_b and shape_c with shape_d
    match_events = [(0, 0, slice(10, 20), 'shape_a', 10), (0, 0, slice(9, 21), 'shape_b', 10),
                    (1, 0, slice(9, 21), 'shape_b', 10), (2, 0, slice(10, 20), 'shape_a', 10),
                    (0, 0, slice(30, 40), 'shape_c', 10), (2, 0, slice(20, 30), 'shape_e', 10),
                    (0, 0, slice(29, 31), 'shape_d', 10)]

    # The first and last event should be removed if we prioritize shape_b over shape_a and shape_c over shape_d
    answer = [(0, 0, slice(9, 21), 'shape_b', 10), (1, 0, slice(9, 21), 'shape_b', 10),
              (2, 0, slice(10, 20), 'shape_a', 10), (0, 0, slice(30, 40), 'shape_c', 10),
              (2, 0, slice(20, 30), 'shape_e', 10)]

    test_output = util.red_event_sort(match_events, [('shape_b', 'shape_a'), ('shape_c', 'shape_d')])
    assert test_output == answer, "The events were not sorted properly"


def test_match_fraction():
    # Make up a simple event list belonging to some fictitious data with 5 times and 100 frequencies
    events = [(1, slice(0, 10), 'shape_1', 10),
              (2, slice(0, 10), 'shape_1', 10),
              (3, slice(10, 20), 'shape_2', 10),
              (4, slice(30, 30), 'narrow', 10)]
    Ntimes = 5
    Nfreqs = 100
    # Make the event_fraction dictionary
    event_frac = util.event_fraction(events, Ntimes, ['shape_1', 'shape_2'], Nfreqs)

    assert event_frac == {'shape_1': 2 / 5, 'shape_2': 1 / 5, 'narrow': 1 / (Ntimes * Nfreqs)}, "Event fraction not calculated correctly"
