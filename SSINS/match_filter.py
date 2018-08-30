from __future__ import absolute_import, division, print_function

import numpy as np
from SSINS import util
from scipy.special import erfc


class MF:

    def __init__(self, INS, sig_thresh=None, shape_dict={}, N_thresh=0, alpha=None,
                 point=True, streak=True):
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

    def apply_match_test(self):

        count = 1
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
            self.INS.data_ms = self.INS.mean_subtract()
        self.INS.counts, self.INS.bins, self.INS.sig_thresh = self.INS.hist_make(sig_thresh=self.sig_thresh)

    def apply_chisq_test(self):
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

            self.INS.data_ms = self.INS.mean_subtract()
        self.INS.counts, self.INS.bins, self.INS.sig_thresh = self.INS.hist_make(sig_thresh=self.sig_thresh)

    def apply_samp_thresh_test(self):

        assert self.N_thresh < self.INS.data.shape[0], 'N_thresh is greater than the number of times. This would result in flagging the entire observation regardless of content.'
        ind = np.where(np.count_nonzero(np.logical_not(self.INS.data.mask), axis=0) < self.N_thresh)[:-1]
        self.INS.samp_thresh_events = np.vstack(ind).T
        self.INS.data[:, ind[0], ind[1]] = np.ma.masked
        self.INS.data_ms = self.INS.mean_subtract()
        self.INS.counts, self.INS.bins, self.INS.sig_thresh = self.INS.hist_make(sig_thresh=self.sig_thresh)
