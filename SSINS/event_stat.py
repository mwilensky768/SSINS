"""
Class for more deeply examining events that were identified in INS.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import warnings
import os
import pickle
import time


class ES:

    """
    ES stands for Event Statistics. This is really just a data container for
    events that are caught by the match filter.
    """

    def __init__(self, flag_choice=None, outpath=None, obs=None,
                 read_paths={}, match_events=None,
                 match_hists=None, chisq_events=None, chisq_hists=None,
                 samp_thresh_events=None):

        """
        init function for ES class. Attributes can be read in from numpy loadable
        files using the read_paths dictionary keyword, otherwise they can be set
        manually.

        Args:
            read_paths (dict): A dictionary whose keys are the attributes to be
                               read in, and whose values are paths to files
                               containing the data for those attributes.
            flag_choice (str): See attributes. Thus must be set manually.
            obs (str): See attributes. Thus must be set manually.
            outpath (str): See attributes. Thus must be set manually.
            **attributes: The other attributes can be set manually with keywords.
        """

        self.flag_choice = flag_choice
        """The choice of flagging in the SS object."""
        self.outpath = outpath
        """The directory to save outputs."""
        self.obs = obs
        """The obsid in question"""

        if not read_paths:
            if match_events is None:
                self.match_events = []
                """A list of events caught by the match_test"""
            else:
                self.match_events = match_events
            if match_hists is None:
                self.match_hists = []
                """A list of histograms of match_event-averaged mean-subtracted INS data"""
            else:
                self.match_hists = match_hists
            if chisq_events is None:
                self.chisq_events = []
                """A list of events caught by the chisq_test"""
            else:
                self.chisq_events = chisq_events
            if chisq_hists is None:
                self.chisq_hists = []
                """A list of histograms of chisq_event-averged mean-subtracted INS data"""
            else:
                self.chisq_hists = chisq_hists
            if self.samp_thresh_events is None:
                self.samp_thresh_events = []
                """A list of events found by the samp_thresh_test"""
            else:
                self.samp_thresh_events = semp_thresh_events

        else:
            self._read(read_paths)

    def save(self):
        """
        Writes out the attributes of this class to ES.outpath using pickle.
        """
        for attr in ['obs', 'outpath']:
            if getattr(self, attr) is None:
                raise TypeError("To save data, ES.%s should not be None" % attr)

        if self.flag_choice is None:
            warnings.warn("Saving data with ES.flag_choice as None.")

        path = '%s/arrs' % outpath
        if not os.path.exists(path):
            os.makedirs(path)

        attrs = ['match_events', 'match_hists', 'chisq_events', 'chisq_hists',
                 'samp_thresh_events']
        for attr in attrs:
            with open('%s/%s_%s_%s.pik' % (path, obs, flag_choice, attr), 'wb') as file:
                pickle.dump(getattr(self, attr), file)

    def _read(self, read_paths):
        """
        Reads in the attributes which are present in the read_paths dictionary.

        Args:
            read_paths (dict): The keys of this dictionary should be the
                               attributes to read in, while the values should be
                               paths to files containing the information for
                               those attributes.
        """
        attrs = ['match_events', 'match_hists', 'chisq_events', 'chisq_hists',
                 'samp_thresh_events']
        for attr in attrs:
            if attr in read_paths:
                with open(read_paths[attr], 'rb') as file:
                    setattr(self, attr, pickle.load(file))
            else:
                setattr(self, attr, [])

    def hist_make(self, INS, bins=None, event=None):

        """
        A function which will make histograms of the mean-subtracted data.

        Args:
            sig_thresh: The significance threshold within which to make the
                        primary bins. Bins are of unit width. Data outside the
                        sig_thresh will be placed in a single outlier bin.
                        Will calculate a reasonable one by default.
            event: Used to histogram a single shape or frequency channel.
                   Providing an event as found in the INS.match_events list
                   will histogram the data corresponding to the shape for that
                   event, where the data is averaged over that shape before
                   histogramming.

        Returns:
            counts (array): The counts in each bin.
            bins (array): The bin edges.
            sig_thresh (float): The sig_thresh parameter. Will be the calculated value
                                if sig_thresh is None, else it will be what sig_thresh
                                was set to.
        """

        if bins is None:
            bin_max = np.sqrt(2) * erfcinv(1. / np.prod(INS.data.shape))
        bins = np.linspace(-bin_max, bin_max,
                           num=int(2 * np.ceil(2 * sig_thresh)))
        if event is None:
            dat = INS.metric_ms
        else:
            N = np.count_nonzero(np.logical_not(INS.data_ms.mask[:, 0, event[2]]), axis=1)
            dat = INS.metric_ms[:, 0, event[2]].mean(axis=1) * np.sqrt(N)
        if dat.min() < -sig_thresh:
            bins = np.insert(bins, 0, dat.min())
        if dat.max() > sig_thresh:
            bins = np.append(bins, dat.max())
        counts, _ = np.histogram(dat[np.logical_not(dat.mask)],
                                 bins=bins)

        return(counts, bins)
