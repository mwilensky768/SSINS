from __future__ import absolute_import, division, print_function

import scipy.stats
import numpy as np


def hist_fit(counts, bins, dist='norm'):
    N = np.sum(counts)
    p = getattr(scipy.stats, dist).cdf(bins[1:]) - getattr(scipy.stats, dist).cdf(bins[:-1])
    exp = N * p
    var = N * p * (1 - p)

    return(exp, var)


def bin_combine(counts, bins, weight='var', thresh=1, dist='norm'):

    exp, var = hist_fit(counts, bins, dist=dist)

    if weight is 'exp':
        # Sum the counts and make sure a valid bin is possible at all
        S = np.sum(counts)
        c_cond = np.logical_or(exp < thresh, counts < thresh)
    elif weight is 'var':
        # Sum the var and make sure a valid bin is possible at all
        S = np.sum(var)
        c_cond = var < thresh
    if S > thresh:
        while np.any(c_cond) and len(counts) > 2:
            counts[1] += counts[0]
            counts[-2] += counts[-1]
            counts = counts[1:-1]
            bins = np.delete(bins, (1, len(bins) - 2))
            exp, var = hist_fit(counts, bins, dist=dist)
            if weight is 'exp':
                c_cond = np.logical_or(exp < thresh, counts < thresh)
            elif weight is 'var':
                c_cond = var < thresh

    return(counts, bins)


def chisq(counts, bins, weight='var', thresh=1, dist='norm'):

    counts, bins = bin_combine(counts, bins, weight=weight, thresh=thresh, dist=dist)
    exp, var = hist_fit(counts, bins, dist=dist)
    if weight is 'exp':
        S = np.sum(counts)
        stat, p = scipy.stats.chisquare(counts, exp, ddof=2)
    elif weight is 'var':
        S = np.sum(var)
        stat = np.sum((counts - exp)**2 / var)
        p = scipy.stats.chi2.isf(stat, len(var) - 3)
    if S < thresh:
        stat, p = np.nan, np.nan

    return(stat, p)


def slc_len(slc, shape):
    return(slc.indices(shape)[1] - slc.indices(shape)[0])


def event_fraction(match_events, Nfreqs, Ntimes):

    print(np.array(match_events))
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
    with open(obsfile) as f:
        obslist = f.read().split("\n")
    while '' in obslist:
        obslist.remove('')
    obslist.sort()
    return(obslist)
