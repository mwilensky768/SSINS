"""
Some utility functions. Some are utilized within the main code and some are just
nice to have around.
"""

from __future__ import absolute_import, division, print_function

import scipy.stats
import numpy as np
import os
from astropy.io import fits
from functools import reduce


def red_event_sort(match_events, shape_tuples):

    """
    This is a function that is used to filter redundant events from a match_events
    list, in case events with different names (or even shapes) are physically caused
    by the same mechanism.

    Args:
        match_events: A list of match_events from an INS found by a match filter
        shape_tuples:
            A list of tuples where each tuple indicates which events are
            considered redundant with one another. Each tuple must be ordered
            according to the priority in which the shapes are to be kept.

    Returns:
        match_events: The filtered match_event list.
    """

    removal_events = []
    # Iterate over the redundancy sets
    for shape_tup in shape_tuples:
        # Filter the shapes that are in the current shape tuple
        relevant_events = [event for event in match_events if event[-2] in shape_tup]
        all_times = [event[0] for event in relevant_events]
        # Generate an array of unique times and counts
        unique_times, counts = np.unique(all_times, return_counts=True)
        # find times with redundancies
        red_times = unique_times[counts > 1]
        # iterate over the times with redundancies
        for time in unique_times:
            # Collect relevant events that occurred at each time
            red_events = [event for event in relevant_events if event[0] == time]
            # Sort them according to the priority in the shape tuple
            sorted_red_events = sorted(red_events, key=lambda x: shape_tup.index(x[-2]))
            # Add the tail of the list to the removal_list
            removal_events += sorted_red_events[1:]

    for event in removal_events:
        match_events.remove(event)

    return(match_events)


def event_fraction(match_events, Ntimes, shape_list, Nfreqs=None):
    """
    Calculates the fraction of times an event was caught by the flagger for
    each type of event. Narrowband occupancy levels are calculated over the whole
    spectrum, instead of just the channel a particular event may have belonged to.

    Args:
        match_events; A list of events caught by the match_filter
        Ntimes: The number of time integrations in the observation
        shape_list: The list of shapes to calculate the occupation fractions for
        Nfreqs: Number of frequencies in the observation. Only required if looking for narrow occupancy levels.

    Returns:
        match_event_frac: A dictionary with shapes for keys and occupancy fractions for values
    """
    match_event_frac = {shape: 0 for shape in shape_list}

    if match_events:
        shapes, counts = np.unique(np.array(match_events)[:, -2], return_counts=True)
        for i, shape in enumerate(shapes):
            if shape is 'narrow':
                match_event_frac[shape] = counts[i] / (Ntimes * Nfreqs)
            else:
                match_event_frac[shape] = counts[i] / Ntimes

    return(match_event_frac)


def make_obslist(obsfile):
    """
    Makes a python list from a text file whose lines are separated by "\n"

    Args:
        obsfile: A text file with an obsid on each line

    Returns:
        obslist: A list whose entries are obsids
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

    Args:
        obslist: A list of obsids
        outpath: The filename to write to
    """
    with open(outpath, 'w') as f:
        for obs in obslist:
            f.write("%s\n" % obs)
