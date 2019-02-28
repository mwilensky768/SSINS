"""
Match Filter class
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from SSINS import util, ES
from scipy.special import erfc
import time


class MF(object):

    """
    Defines the Match Filter (MF) class. It requires an INS object to be passed
    to it in order to operate on it.
    """

    def __init__(self, freq_array, sig_thresh, shape_dict={}, N_thresh=0, alpha=None,
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
        if (not shape_dict) and (not point) and (not streak):
            raise ValueError("There are not shapes in the shape_dict and point/streak shapes are disabled. Check keywords.")

        self.freq_array = freq_array
        self.shape_dict = shape_dict
        self.N_thresh = N_thresh
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
            if min(self.freq_array[0, :]) < min(self.shape_dict[shape]) or \
               max(self.freq_array[0, :]) > max(self.shape_dict[shape]):
                min_chan = np.argmin(np.abs(self.freq_array[0, :] - min(self.shape_dict[shape])))
                max_chan = np.argmin(np.abs(self.freq_array[0, :] - max(self.shape_dict[shape])))
                # May have to extend the edges depending on if the shape extends beyond the min and max chan infinitesimally
                if (self.freq_array[0, min_chan] - min(self.shape_dict[shape]) > 0) and (min_chan > 0):
                    min_chan -= 1
                if self.freq_array[0, max_chan] - max(self.shape_dict[shape]) < 0:
                    max_chan += 1
                slice_dict[shape] = slice(min_chan, max_chan)
        if point:
            slice_dict['point'] = None
        if streak:
            slice_dict['streak'] = slice(None)

        return(slice_dict)

    def match_test(self, INS):

        """
        The primary test that the filter is used with. The greatest outlier in
        each shape is put forth and a champion among them is chosen. The time,
        frequencies, and outlier statistic and shape of this champion is
        returned to the stack.
        """

        R_max = -np.inf
        t_max = None
        f_max = None
        shape_max = None
        for shape in self.slice_dict:
            if shape is 'point':
                t, f, p = np.unravel_index(np.absolute(INS.metric_ms).argmax(),
                                           INS.metric_ms.shape)
                R = np.absolute(INS.metric_ms[t, f, p] / self.sig_thresh)
                f = slice(f, f + 1)
            else:
                N = np.count_nonzero(np.logical_not(INS.metric_ms[:, self.slice_dict[shape]].mask),
                                     axis=1)
                sliced_arr = np.absolute(INS.metric_ms[:, self.slice_dict[shape]].mean(axis=1)) * np.sqrt(N)
                t, p = np.unravel_index((sliced_arr / self.sig_thresh).argmax(),
                                        sliced_arr.shape)
                f = self.slice_dict[shape]
                R = sliced_arr[t, p] / self.sig_thresh
            if R > 1:
                if R > R_max:
                    t_max, f_max, R_max, shape_max = (t, f, R, shape)
        return(t_max, f_max, R_max, shape_max)

    def apply_match_test(self, INS, es=None, event_record=False,
                         apply_N_thresh=False):

        """
        Where match_test() is implemented. The champion from match_test() is
        flagged if its outlier statistic is greater than sig_thresh, and the
        mean-subtracted spectrum is recalculated. This repeats
        until there are no more outliers greater than sig_thresh. Also can apply
        the samp_thresh test, which flags channels between match test iterations
        if those channels have less than N_thresh unflagged samples left.
        """
        print('Beginning match_test at %s' % time.strftime("%H:%M:%S"))

        if event_record and (es is None):
            es = ES()

        count = 1
        while count:
            count = 0
            t_max, f_max, R_max, shape_max = self.match_test(INS)
            if R_max > -np.inf:
                count += 1
                event = (t_max, f_max, shape_max)
                INS.metric_array[event[:-1]] = np.ma.masked
                if event_record:
                    es.match_events.append(event)
                if (apply_N_thresh and self.N_thresh):
                    self.apply_samp_thresh_test(INS, es=es, event_record=event_record)
                if not np.all(INS.metric_array[:, f_max, 0].mask):
                    INS.metric_ms[:, f_max] = INS.mean_subtract(f=f_max)
                else:
                    INS.metric_ms[:, f_max] = np.ma.masked

        return(es)

        print('Finished match_test at %s' % time.strftime("%H:%M:%S"))

    def apply_samp_thresh_test(self, INS, es=None, event_record=False):
        """
        The sample threshold test. This tests to see if any channels have fewer
        unflagged channels than a given threshold. If so, the entire channel is
        flagged. A ValueError is raised if the threshold parameter is greater
        than the number of times in the observation, due to the fact that this
        will always lead to flagging the entire observation.
        """

        if self.N_thresh > INS.metric_array.shape[0]:
            raise ValueError("N_thresh parameter is set higher than "
                             "the number of time samples. This will "
                             "always result in flagging the entire "
                             "observation. Aborting flagging.")
        good_chans = np.where(np.logical_not(np.all(INS.metric_array[:, 0, :, 0].mask, axis=0)))[0]
        N_unflagged = INS.metric_array.shape[0] - np.count_nonzero(INS.metric_array.mask[:, 0, good_chans, 0], axis=0)
        if np.any(N_unflagged < self.N_thresh):
            chans = np.where(N_unflagged < self.N_thresh)[0]
            if event_record:
                es.samp_thresh_events.append(good_chans[chans])
            INS.data[:, 0, good_chans[chans]] = np.ma.masked
