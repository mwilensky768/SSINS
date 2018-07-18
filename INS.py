import numpy as np
from scipy.special import erfcinv
import os


class Spectrum:
    """
    This incoherent noise spectrum class, formed from an RFI object as an input.
    Outputs a data array
    """

    def __init__(self, data, Nbls, freq_array, pols, vis_units, obs, outpath,
                 match_events=[], match_hists=[], chisq_events=[], chisq_hists=[]):

        args = (data, Nbls, freq_array, pols, vis_units, obs, outpath)
        kwds = ('data', 'Nbls', 'freq_array', 'pols', 'vis_units', 'obs', 'outpath')
        assert all([arg is not None for arg in args]), \
            'Insufficient input given. Supply an instance of the RFI class or read in appropriate data.'
        kwargs = dict(zip(kwds, args))
        for kwd in kwds:
            setattr(self, kwd, kwarg[kwd])

        self.data_ms = self.mean_subtract()
        self.counts, self.bins, self.sig_thresh = self.hist_make()
        self.match_events = match_events
        self.match_hists = match_hists
        self.chisq_events = chisq_events
        self.chisq_hists = chisq_hists
        for string in ['arrs', 'figs']:
            if not os.path.exists('%s/%s' % ('arrs', 'figs')):
                os.makedirs('%s/%s' % ('arrs', 'figs'))

    def mean_subtract(self):

        C = 4 / np.pi - 1
        MS = (self.data / self.data.mean(axis=0) - 1) * np.sqrt(self.Nbls / C)
        return(MS)

    def hist_make(self, sig_thresh=None, event=None):

        if sig_thresh is None:
            sig_thresh = np.sqrt(2) * erfcinv(1. / np.prod(self.data.shape))
        bins = np.linspace(-self.sig_thresh, self.sig_thresh,
                           num=int(2 * np.ceil(2 * self.sig_thresh)))
        if event is None:
            dat = self.data_ms
        else:
            N = np.count_nonzero(np.logical_not(self.data_ms.mask[:, event[0], event[1]]), axis=1)
            dat = self.data_ms[:, event[0], event[1]].mean(axis=1) * np.sqrt(N)
        if dat.min() < -sig_thresh:
            bins = np.insert(bins, 0, dat.min())
        if dat.max() > sig_thresh:
            bins = np.append(bins, dat.max())
        counts, _ = np.histogram(dat[np.logical_not(dat.mask)],
                                 bins=bins)

        return(counts, bins, sig_thresh)
