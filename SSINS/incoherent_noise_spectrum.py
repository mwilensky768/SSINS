from __future__ import absolute_import, division, print_function

"""
The incoherent noise spectrum class.
"""

import numpy as np
from scipy.special import erfcinv
import os
import warnings
import pickle


class INS:
    """
    Defines the incoherent noise spectrum (INS) class.
    """

    def __init__(self, data=None, Nbls=None, freq_array=None, pols=None,
                 flag_choice=None, vis_units=None, obs=None, outpath=None,
                 match_events=[], match_hists=[], chisq_events=[],
                 chisq_hists=[], read_paths={}, samp_thresh_events=[],
                 order=0):

        """
        init function for the INS class. Can set the attributes manually, or
        read some in using the read_paths dictionary. The keys for read_paths
        are the attribute names as strings, while the values are paths to
        numpy loadable binary files (pickle is used for masked arrays). The init
        function will calculate the Calculated Attributes (see below).

        Required Parameters: These parameters must either be passed manually or
                             included in the read_paths dictionary.

                             data: The baseline-averaged sky-subtracted
                                   visibility amplitudes.

                             Nbls: An array of the same shape as data which
                                   tells how many baselines were used for each
                                   point in data.

                             freq_array: frequencies for the data

        Optional Attributes: pols: The polarizations present in the data.

                             flag_choice: The type of flags used in the
                                          sky_subtract object before
                                          averaging across the baselines

                             vis_units: The units for the visibilities

                             obs: The OBSID for the spectrum

                             outpath: A base directory to save attributes to

         Match Filter Attributes: These attributes are not typically passed, but
                                  instead calculated using the MF class.

                                  match_events: Events (time/frequency-slice
                                                pairs) in the spectrum located
                                                by the match filter.

                                  match_hists: Histograms for the match events
                                               above.

                                  chisq_events: Channels or events which were
                                                identified by the chisq test in
                                                the MF class.

                                 chisq_hists: Histograms for the above events

                                 samp_thresh_events: Events located by the
                                                     samp thresh test in the MF
                                                     class.

         Calculated Attributes: The attributes are calculable using
                                mean_subtract() and hist_make().

                                data_ms: The mean-subtracted data array. The
                                         data in this array is standardized
                                         according to an estimator of the noise.

                                counts:  The counts for the binned
                                         mean-subtracted data

                                bins: The bins for the mean-subtracted data

                                sig_thresh: Without employing the match filter,
                                            this parameter just describes the
                                            default bins for the mean-subtracted
                                            data. If a MF class is initialized
                                            without passing the sig_thresh kwarg,
                                            then the INS.sig_thresh is used for
                                            the match filter as well. The
                                            default calculation is the same for
                                            both classes.
         Keywords: order: The order of the mean_subtract function on
                          initialization. For now, can possible allow for a
                          linear drift in the mean (order=1).
        """

        opt_args = {'obs': obs, 'pols': pols, 'vis_units': vis_units,
                    'outpath': outpath, 'flag_choice': flag_choice}
        for attr in ['obs', 'outpath']:
            if opt_args[attr] is None:
                warnings.warn('In order to save outputs and use Catalog.py, \
                               please supply %s attribute' % (attr))
            else:
                setattr(self, attr, opt_args[attr])

        if flag_choice is None:
            warnings.warn('%s%s%s%s' % ('flag_choice is set to None. If this ',
                                        'does not reflect the flag_choice of ',
                                        'the original data, then saved arrays ',
                                        'will be mislabled'))
        self.flag_choice = flag_choice

        if not read_paths:
            args = (data, Nbls, freq_array)
            assert all([arg is not None for arg in args]),\
                '%s%s%s' % ('Insufficient data given. You must supply a data',
                            ' array, a Nbls array of matching shape, and a freq',
                            '_array of matching sub-shape')
            self.data = data
            self.Nbls = Nbls
            self.freq_array = freq_array

            for attr in ['pols', 'vis_units']:
                if opt_args[attr] is None:
                    warnings.warn('%s%s%s' % ('In order to use Catalog_Plot.py',
                                              ' with appropriate labels, please',
                                              ' supply %s attribute' % (attr)))
                else:
                    setattr(self, attr, opt_args[attr])

            kwargs = {'match_events': match_events, 'match_hists': match_hists,
                      'chisq_events': chisq_events, 'chisq_hists': chisq_hists,
                      'samp_thresh_events': samp_thresh_events}
            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])
        else:
            self.read(read_paths)

        self.data_ms = self.mean_subtract(order=order)
        self.counts, self.bins, self.sig_thresh = self.hist_make()

    def mean_subtract(self, f=slice(None), order=0):

        """
        A function which calculated the mean-subtracted spectrum from the
        regular spectrum. A spectrum made from a perfectly clean observation
        will be standardized (written as a z-score) by this operation. Setting
        order=1 allows for a linear drift in the mean w/respect to time.
        """

        # This constant is determined by the Rayleigh distribution, which
        # describes the ratio of its rms to its mean
        C = 4 / np.pi - 1
        if not order:
            MS = (self.data[:, :, f] / self.data[:, :, f].mean(axis=0) - 1) * np.sqrt(self.Nbls[:, :, f] / C)
        else:
            MS = np.ma.masked_array(np.zeros(self.data[:, :, f].shape))
            x = np.arange(self.data.shape[0])
            for i in range(self.data.shape[-1]):
                y = self.data[:, 0, f, i]
                coeff = np.ma.polyfit(x, y, order)
                mu = np.sum([np.outer(x**(order - k), coeff[k]) for k in range(order + 1)], axis=0)
                MS[:, 0, :, i] = (self.data[:, 0, f, i] / mu - 1) * np.sqrt(self.Nbls[:, 0, f, i] / C)

        return(MS)

    def hist_make(self, sig_thresh=None, event=None):

        """
        A function which will make histograms. Bins can me modulated using the
        sig_thresh parameter, which is related to the sig_thresh paramter of the
        match filter (MF) class. A reasonable sig_thresh can be determined
        from the size of the data set by looking for the z-score beyond which
        less than 1 count of noise is expected. If an event is given, then
        data is averaged over the frequencies of the event before construction
        of the histogram.
        """

        if sig_thresh is None:
            sig_thresh = np.sqrt(2) * erfcinv(1. / np.prod(self.data.shape))
        bins = np.linspace(-sig_thresh, sig_thresh,
                           num=int(2 * np.ceil(2 * sig_thresh)))
        if event is None:
            dat = self.data_ms
        else:
            N = np.count_nonzero(np.logical_not(self.data_ms.mask[:, 0, event[2]]), axis=1)
            dat = self.data_ms[:, 0, event[2]].mean(axis=1) * np.sqrt(N)
        if dat.min() < -sig_thresh:
            bins = np.insert(bins, 0, dat.min())
        if dat.max() > sig_thresh:
            bins = np.append(bins, dat.max())
        counts, _ = np.histogram(dat[np.logical_not(dat.mask)],
                                 bins=bins)

        return(counts, bins, sig_thresh)

    def save(self):
        """
        Writes out relevant data products.
        """
        tags = ['match', 'chisq', 'samp_thresh']
        tag = ''
        for subtag in tags:
            if len(getattr(self, '%s_events' % (subtag))):
                tag += '_%s' % subtag

        for string in ['arrs', 'metadata']:
            if not os.path.exists('%s/%s' % (self.outpath, string)):
                os.makedirs('%s/%s' % (self.outpath, string))

        for attr in ['data', 'data_ms', 'Nbls']:
            with open('%s/arrs/%s_%s_INS_%s%s.npym' %
                      (self.outpath, self.obs, self.flag_choice, attr, tag), 'wb') as f:
                pickle.dump(getattr(self, attr), f)
        for attr in ['counts', 'bins']:
            np.save('%s/arrs/%s_%s_INS_%s%s.npy' %
                    (self.outpath, self.obs, self.flag_choice, attr, tag),
                    getattr(self, attr))
        for attr in ['match_events', 'match_hists', 'chisq_events',
                     'chisq_hists', 'samp_thresh_events']:
            if len(getattr(self, attr)):
                np.save('%s/arrs/%s_%s_INS_%s.npy' %
                        (self.outpath, self.obs, self.flag_choice, attr),
                        getattr(self, attr))

        for attr in ['freq_array', 'pols', 'vis_units']:
            if hasattr(self, attr):
                np.save('%s/metadata/%s_%s.npy' % (self.outpath, self.obs, attr),
                        getattr(self, attr))

    def read(self, read_paths):
        """
        Reads in attributes from numpy loadable (or unpicklable) files whose
        paths are specified in a read_paths dictionary, where the keys are the
        attributes and the values are the paths.
        """

        for arg in ['data', 'Nbls', 'freq_array']:
            assert arg in read_paths,\
                'You must supply a path to a numpy loadable %s file for read_paths entry' % (arg)
            setattr(self, arg, np.load(read_paths[arg]))
        for attr in ['pols', 'vis_units']:
            if attr not in read_paths:
                warnings.warn('In order to use Catalog_Plot.py, please supply\
                               path to numpy loadable %s attribute for read_paths entry' % (attr))
            else:
                setattr(self, attr, np.load(read_paths[attr]))
        for attr in ['match', 'chisq']:
            for subattr in ['events', 'hists']:
                attribute = '%s_%s' % (attr, subattr)
                if attribute in read_paths:
                    setattr(self, attribute, list(np.load(read_paths[attribute])))
                else:
                    setattr(self, attribute, [])
        if 'samp_thresh_events' in read_paths:
            self.samp_thresh_events = np.load(read_paths['samp_thresh_events'])
        else:
            self.samp_thresh_events = []
