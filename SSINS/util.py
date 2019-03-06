"""
Some utility functions. Some are utilized within the main code and some are just
nice to have around given this set of code.
"""

from __future__ import absolute_import, division, print_function

import scipy.stats
import numpy as np
import os
from astropy.io import fits
from functools import reduce


def slc_len(slc, shape):
    """
    Returns the length of a list of indices that describe a slice object.
    """
    return(slc.indices(shape)[1] - slc.indices(shape)[0])


def red_event_sort(match_events, shape_tuples, keep_prior=[1, 0]):
    """
    This is a function that is used to filter redundant events from a match_events
    list, in case events with different names (or even shapes) are physically caused
    by the same mechanism.
    """
    removal_list = []
    # iterate through the shape pairs
    for tup in shape_tuples:
        # Separate the events belonging to each member of the shape_tuple
        event_time_lists = [[event[0] for event in match_events if event[-1] == tup[i]] for i in range(len(tup))]
        all_times, counts = np.unique(reduce((lambda x, y: x + y), event_time_lists), return_counts=True)
        for event_time, count in zip(all_times, counts):
            if count > 0:
                list_ind = []
                for i in range(len(tup)):
                    if event_time in event_time_lists[i]:
                        list_ind.append(i)
                # Returns the list of indices for event_time_lists where the event must be removed
                removal_list_inds = sorted(list_ind, key=lambda x: keep_prior.index(x))[1:]
                for i in removal_list_inds:
                    removal_list.append((event_time, tup[i]))
    match_events = [event for event in match_events if (event[0], event[-1]) not in removal_list]

    return(match_events)


def event_fraction(match_events, Ntimes, shape_list, Nfreqs):
    """
    Calculates the fraction of times an event was caught by the flagger for
    each type of event.
    """
    shapes, counts = np.unique(np.array(match_events)[:, -1], return_counts=True)
    match_event_frac = {shape: 0 for shape in shape_list}
    for i, shape in enumerate(shapes):
        if shape is 'point':
            match_event_frac[shape] = counts[i] / (Ntimes * Nfreqs)
        else:
            match_event_frac[shape] = counts[i] / Ntimes

    return(match_event_frac)


def make_obslist(obsfile):
    """
    Makes a python list from a text file whose lines are separated by "\n"
    """
    with open(obsfile) as f:
        obslist = f.read().split("\n")
    while '' in obslist:
        obslist.remove('')
    obslist.sort()
    return(obslist)


def make_obsfile(obslist, outpath):
    """
    Makes a text file from a list of obsids
    """
    with open(outpath, 'w') as f:
        for obs in obslist:
            f.write("%s\n" % obs)
