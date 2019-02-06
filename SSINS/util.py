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


def hist_fit(counts, bins, dist='norm'):
    """
    Given a histogram, draws the expected counts and variance for the specified
    distribution. Must be a scipy distribution.
    """
    N = np.sum(counts)
    p = getattr(scipy.stats, dist).cdf(bins[1:]) - getattr(scipy.stats, dist).cdf(bins[:-1])
    exp = N * p
    var = N * p * (1 - p)

    return(exp, var)


def bin_combine(counts, bins, weight='var', thresh=1, dist='norm'):

    """
    Combines bins from the outside in until all bins have a weight which exceeds
    thresh. Used for making a reasonable chisquare calculation.

    Arguments: counts: The counts in the histogram bins

               bins: The bin edges for the histogram

               weight: Choices are 'var' or 'exp'

                       'var': The expected variance in each bin must exceed
                              thresh

                       'exp': The expected counts AND the counts must exceed
                              thresh in each bin

               dist: The scipy distribution to calculate expected counts/bins
                      with
    """

    exp, var = hist_fit(counts, bins, dist=dist)
    c_com, b_com = (np.copy(counts), np.copy(bins))

    if weight is 'exp':
        # Sum the counts and make sure a valid bin is possible at all
        S = np.sum(counts)
        c_cond = np.logical_or(exp < thresh, counts < thresh)
    elif weight is 'var':
        # Sum the var and make sure a valid bin is possible at all
        S = np.sum(var)
        c_cond = var < thresh
    if S > thresh:
        while np.any(c_cond) and len(c_com) > 4:
            c_com[1] += c_com[0]
            c_com[-2] += c_com[-1]
            c_com = c_com[1:-1]
            b_com = np.delete(b_com, (1, len(b_com) - 2))
            exp, var = hist_fit(c_com, b_com, dist=dist)
            if weight is 'exp':
                c_cond = np.logical_or(exp < thresh, c_com < thresh)
            elif weight is 'var':
                c_cond = var < thresh

    return(c_com, b_com)


def chisq(counts, bins, weight='var', thresh=1, dist='norm'):

    """
    Calculates a chisq statistic given a distribution and a chosen weight
    scheme.
    """

    counts, bins = bin_combine(counts, bins, weight=weight, thresh=thresh, dist=dist)
    exp, var = hist_fit(counts, bins, dist=dist)
    if weight is 'exp':
        S = np.sum(counts)
        stat, p = scipy.stats.chisquare(counts, exp, ddof=2)
    elif weight is 'var':
        S = np.sum(var)
        stat = np.sum((counts - exp)**2 / var)
        p = scipy.stats.chi2.sf(stat, len(var) - 3)
    if S < thresh:
        stat, p = np.nan, np.nan

    return(stat, p)


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


def read_paths_construct(basedir, flag_choice, obs, product, tag='',
                         exclude=None):

    """
    Makes a read_paths dictionary from data which was saved using the save()
    functions of the various objects.
    """

    read_paths = {}

    if product is 'INS':
        mask_attrs = ['data', 'Nbls']
        attrs = ['match_events', 'match_hists', 'chisq_events', 'chisq_hists',
                 'samp_thresh_events']

    if product is 'VDH':
        mask_attrs = ['W_hist', 'MLEs']
        attrs = ['counts', 'bins', 'fits', 'errors']

    if product is 'ES':
        mask_attrs = ['avgs', 'uv_grid']
        attrs = ['counts', 'exp_counts', 'exp_error', 'bins', 'cutoffs']
        path = '%s/metadata/%s_grid.npy' % (basedir, obs)
        if os.path.exists(path):
            read_paths['grid'] = path

    for attr in mask_attrs:
        path = '%s/arrs/%s_%s_%s_%s%s.npym' % (basedir, obs, flag_choice,
                                               product, attr, tag)
        if os.path.exists(path):
            read_paths[attr] = path
    for attr in attrs:
        path = '%s/arrs/%s_%s_%s_%s.npy' % (basedir, obs, flag_choice, product,
                                            attr)
        if os.path.exists(path):
            read_paths[attr] = path

    for attr in ['freq_array', 'pols', 'vis_units']:
        path = '%s/metadata/%s_%s.npy' % (basedir, obs, attr)
        if os.path.exists(path):
            read_paths[attr] = path

    if exclude is not None:
        for attr in exclude:
            read_paths.pop(attr)

    return(read_paths)
