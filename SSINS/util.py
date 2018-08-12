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
        c_cond = np.logical_or(exp < thresh, counts < thresh)
    elif weight is 'var':
        c_cond = var < thresh

    while np.any(c_cond):
        ind = np.where(c_cond)[0][0]
        # If the index is zero, add the bin on the right and delete the bin on
        # the right. Else, add the bin on the left and delete the bin on the left.
        counts[ind] += counts[ind + (-1)**(bool(ind))]
        counts = np.delete(counts, ind + (-1)**(bool(ind)))
        bins = np.delete(bins, ind + (-1)**(bool(ind)))
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
        stat, p = scipy.stats.chisquare(counts, exp, ddof=2)
    elif weight is 'var':
        stat = np.sum((counts - exp)**2 / var)
        p = scipy.stats.chi2.isf(stat, len(var) - 3)

    return(stat, p)


def slc_len(slc, shape):
    return(slc.indices(shape)[1] - slc.indices(shape)[0])


def event_fraction(match_events, Nfreqs, Ntimes):

    shapes, counts = np.unique(np.array(match_events)[:, :-1], return_counts=True)
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
