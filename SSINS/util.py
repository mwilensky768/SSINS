"""
Some utility functions. Some are utilized within the main code and some are just
nice to have around given this set of code.
"""

from __future__ import absolute_import, division, print_function

import scipy.stats
import numpy as np
import os


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


def event_fraction(match_events, Nfreqs, Ntimes):
    """
    Calculates the fraction of times an event was caught by the flagger for
    each type of event.
    """

    shapes, counts = np.unique(np.array(match_events)[:, 2], return_counts=True)
    # Explicit for loop since problem with dict comp involving unhashable types
    keys, values = [], []
    for i, shape in enumerate(shapes):
        if type(shape) is slice:
            keys.append(tuple([shape.indices(Nfreqs)[k] for k in range(2)]))
            values.append(counts[i] / Ntimes)
    match_event_frac = dict(zip(keys, values))

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


def read_paths_INS(basedir, flag_choice, obs, tag='', exclude=None):

    """
    Makes a read_paths dictionary from data which was saved using the save()
    functions of the various objects.
    """

    read_paths = {}
    for attr in ['data', 'Nbls']:
        read_paths[attr] = '%s/arrs/%s_%s_INS_%s%s.npym' % (basedir, obs,
                                                            flag_choice, attr,
                                                            tag)

    for attr in ['freq_array', 'pols', 'vis_units']:
        path = '%s/metadata/%s_%s.npy' % (basedir, obs, attr)
        if os.path.exists(path):
            read_paths[attr] = path

    for attr in ['match', 'chisq']:
        for subattr in ['events', 'hists']:
            attribute = '%s_%s' % (attr, subattr)
            path = '%s/arrs/%s_%s_INS_%s.npy' % (basedir, obs, flag_choice,
                                                 attribute)
            if os.path.exists(path):
                read_paths[attribute] = path
    path = '%s/arrs/%s_%s_INS_samp_thresh_events.npy' % (basedir, obs, flag_choice)
    if os.path.exists(path):
        read_paths['samp_thresh_events'] = path
    if exclude is not None:
        for attr in exclude:
            read_paths.pop(attr)

    return(read_paths)
