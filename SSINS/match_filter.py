"""
Match Filter class
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from SSINS import util
from scipy.special import erfc
import time
from SSINS import Catalog_Plot as cp


class MF:

    """
    Defines the Match Filter (MF) class. It requires an INS object to be passed
    to it in order to operate on it.
    """

    def __init__(self, INS, sig_thresh=None, shape_dict={}, N_thresh=0, alpha=None,
                 point=True, streak=True):

        """
        init function for match filter (MF) class. This function assigns some
        basic attributes and converts the frequencies in the shape_dict to
        channel numbers (see shape_dict description below).

        Parameters: sig_thresh: The outlier threshold in units of the estimated
                                width of the thermal noise. A reasonable
                                choice can be calculated from the size of the
                                data set, by asking for the value beyond which
                                less than 1 count is expected for a data set
                                which is just noise (perfectly clean).

                    shape_dict: This is used in apply_match_test(). The keys are
                                used strictly internally, so they can be named
                                arbitrarily. The values should be the upper and
                                lower frequency limits of the expected RFI
                                (in hz).

                    N_thresh:   Used in apply_samp_thresh_test(). If the number
                                of unflagged samples for a shape or channel is
                                less than N_thresh, then the entire observation
                                will be flagged for that shape or channel.

                    alpha:      Used in apply_chi_square_test(). If a shape or
                                channel's histogram gives a chi-square of p-value
                                less than or equal to alpha, the entire
                                observation will be flagged for that shape
                                or channel.

                    point:      If this is true, single-point outliers will be
                                searched for along with whatever shapes are
                                passed in the shape_dict.

                    streak:     If this is true, broadband RFI which is not
                                band-limited will be sought along with whatever
                                shapes are passed in the shape_dict.
        """

        self.INS = INS
        self.shape_dict = shape_dict
        self.N_thresh = N_thresh
        if sig_thresh is None:
            self.sig_thresh = self.INS.sig_thresh
        else:
            self.sig_thresh = sig_thresh
        if alpha is None:
            self.alpha = erfc(self.sig_thresh / np.sqrt(2))
        self.slice_dict = self.shape_slicer(point, streak)

    def shape_slicer(self, point, streak):

        """
        This function converts the frequency information in the shape_dict
        attribute to slice objects for the channel numbers of the spectrum.
        The point and streak shapes require special slices.
        """

        slice_dict = {}
        for shape in self.shape_dict:
            if min(self.INS.freq_array[0, :]) < min(self.shape_dict[shape]) or \
               max(self.INS.freq_array[0, :]) > max(self.shape_dict[shape]):
                slice_dict[shape] = slice(np.argmin(np.abs(self.INS.freq_array[0, :] - min(self.shape_dict[shape]))),
                                          np.argmin(np.abs(self.INS.freq_array[0, :] - max(self.shape_dict[shape]))))
        if point:
            slice_dict['point'] = None
        if streak:
            slice_dict['streak'] = slice(None)

        return(slice_dict)

    def match_test(self):

        """
        The primary test that the filter is used with. The greatest outlier in
        each shape is put forth and a champion among them is chosen. The time,
        frequencies, and outlier statistic of this champion is returned to the
        stack.
        """

        R_max = -np.inf
        t_max = None
        f_max = None
        for shape in self.slice_dict:
            if shape is 'point':
                t, f, p = np.unravel_index(np.absolute(self.INS.data_ms[:, 0]).argmax(),
                                           self.INS.data_ms[:, 0].shape)
                R = np.absolute(self.INS.data_ms[t, 0, f, p] / self.sig_thresh)
                f = slice(f, f + 1)
            else:
                N = np.count_nonzero(np.logical_not(self.INS.data_ms[:, 0, self.slice_dict[shape]].mask),
                                     axis=1)
                sliced_arr = np.absolute(self.INS.data_ms[:, 0, self.slice_dict[shape]].mean(axis=1)) * np.sqrt(N)
                t, p = np.unravel_index((sliced_arr / self.sig_thresh).argmax(),
                                        sliced_arr.shape)
                f = self.slice_dict[shape]
                R = sliced_arr[t, p] / self.sig_thresh
            if R > 1:
                if R > R_max:
                    t_max, f_max, R_max = (t, f, R)
        return(t_max, f_max, R_max)

    def chisq_test(self):

        """
        A test to measure the chi-square of the binned shapes and channels
        relative to standard normal noise (the null hypothesis of the filter).
        """

        p_min = 1
        shape_min = None
        for shape in self.slice_dict:
            if shape is 'point':
                p = 1
                f_point = None
                for f in range(self.INS.data.shape[2]):
                    stat, p_point = util.chisq(*self.INS.hist_make(sig_thresh=self.sig_thresh,
                                                                   event=(0, 0, slice(f, f + 1)))[:-1])
                    if p_point < p:
                        p = p_point
                        f_point = f
            else:
                stat, p = util.chisq(*self.INS.hist_make(sig_thresh=self.sig_thresh,
                                                         event=(0, 0, self.slice_dict[shape]))[:-1])
            if p < p_min:
                p_min = p
                shape_min = shape

        return(p_min, shape_min, f_point)

    def apply_match_test(self, order=0):

        """
        Where match_test() is implemented. The champion from match_test() is
        flagged if its outlier statistic is greater than sig_thresh, and the
        mean-subtracted spectrum is recalculated. This repeats
        until there are no more outliers greater than sig_thresh.
        """
        print('Beginning match_test at %s' % time.strftime("%H:%M:%S"))

        count = 1
        obs = self.INS.obs
        # Set these attributes to list form so that append method works
        for attr in ['match_events', 'match_hists']:
            if type(getattr(self.INS, attr)) is not list:
                setattr(self.INS, attr, list(getattr(self.INS, attr)))
        while count:
            count = 0
            t_max, f_max, R_max = self.match_test()
            if R_max > -np.inf:
                count += 1
                event = (t_max, 0, f_max)
                self.INS.data[event] = np.ma.masked
                self.INS.match_events.append(event)
                self.INS.match_hists.append(list(self.INS.hist_make(sig_thresh=self.sig_thresh,
                                                                    event=event)))
                self.INS.data_ms[:, :, f_max] = self.INS.mean_subtract(f=f_max,
                                                                       order=order)
        self.INS.counts, self.INS.bins, self.INS.sig_thresh = self.INS.hist_make(sig_thresh=self.sig_thresh)

        print('Finished match_test at %s' % time.strftime("%H:%M:%S"))

    def apply_chisq_test(self):
        """
        Calculates the p-value of each shape and channel using chisq_test().
        Should the p-value of a shape or channel be less than the significance
        threshold, alpha, the entire observation will be flagged for that shape
        or channel.
        """
        # Set these attributes to list so that append method works
        for attr in ['chisq_hists', 'chisq_events']:
            if type(getattr(self.INS, attr)) is not list:
                setattr(self.INS, attr, list(getattr(self.INS), attr))
        p_min = 0
        while p_min < self.alpha:
            p_min, shape_min, f_point = self.chisq_test()
            if p_min < self.alpha:
                if shape_min is 'point':
                    event = (0, 0, slice(f_point, f_point + 1))
                    self.INS.chisq_hists.append(list(self.INS.hist_make(sig_thresh=self.sig_thresh,
                                                                        event=event)))
                    self.INS.data[:, 0, slice(f_point, f_point + 1)] = np.ma.masked
                    self.INS.chisq_events.append(event[2])
                else:
                    event = (0, 0, self.slice_dict[shape_min])
                    self.INS.chisq_hists.append(list(self.INS.hist_make(sig_thresh=self.sig_thresh,
                                                                        event=event)))
                    self.INS.data[:, 0, self.slice_dict[shape_min]] = np.ma.masked
                    self.INS.chisq_events.append(event[2])

                self.INS.data_ms[:, :, event[2]] = self.INS.mean_subtract(f=event[2])
        self.INS.counts, self.INS.bins, self.INS.sig_thresh = self.INS.hist_make(sig_thresh=self.sig_thresh)

    def apply_samp_thresh_test(self):
        """
        A quick test to see if any channels are flagged beyond the tolerance
        described by N_thresh. The goal of this test is to find channels which
        are so thoroughly flagged that their statistics can no longer be trusted,
        in which case it is likely that the entire observation for that channel
        is contaminated, and so that channel is entirely flagged.
        """

        assert self.N_thresh < self.INS.data.shape[0], 'N_thresh is greater than the number of times. This would result in flagging the entire observation regardless of content.'
        ind = np.where(np.count_nonzero(np.logical_not(self.INS.data.mask), axis=0) < self.N_thresh)[:-1]
        self.INS.samp_thresh_events = np.vstack(ind).T
        self.INS.data[:, ind[0], ind[1]] = np.ma.masked
        self.INS.data_ms = self.INS.mean_subtract()
        self.INS.counts, self.INS.bins, self.INS.sig_thresh = self.INS.hist_make(sig_thresh=self.sig_thresh)
