import numpy as np
import Util


class match_filter:

    def __init__(self, INS, sig_thresh=None, shape_dict={}, N_thresh=0, alpha=None):
        self.INS = INS
        self.shape_dict = shape_dict
        self.N_thresh = N_thresh
        if sig_thresh is None:
            self.sig_thresh = self.INS.sig_thresh
        if alpha is None:
            self.alpha = erfcinv(self.sig_thresh / np.sqrt(2))
        self.slice_dict = self.shape_slicer()

    def shape_slicer(self):
        slice_dict = [{} for i in range(self.INS.data.shape[1])]
        for spw in self.INS.data.shape[1]:
            for shape in self.shape_dict:
                if min(self.INS.freq_array[spw, :]) < min(self.shape_dict[shape]) or \
                   max(self.INS.freq_array[spw, :]) > max(self.shape_dict[shape]):
                        slice_dict[spw][shape] = [slice(np.argmin(np.abs(freq_array[spw, :] - min(shape_dict[shape]))),
                                                        np.argmin(np.abs(freq_array[spw, :] - max(shape_dict[shape]))))]
            slice_dict[spw]['streak'] = slice(None)
            slice_dict[spw]['point'] = None

        return(slice_dict)

    def match_test(self, spw):

        R_max = -np.inf
        t_max = None
        f_max = None
        for shape in self.slice_dict[spw]:
            if shape is 'point':
                t, f, p = np.unravel_index(np.absolute(self.INS.data_ms[:, spw]).argmax(), self.INS.data_ms[:, spw].shape)
                R = np.absolute(self.INS.data_ms[t, spw, f, p] / self.sig_thresh)
                f = slice(f, f + 1)
            else:
                N = np.count_nonzero(np.logical_not(self.INS.data_ms[:, spw, slice_dict[spw][shape]].mask), axis=1)
                sliced_arr = np.absolute(self.INS.data_ms[:, spw, slice_dict[spw][shape]].mean(axis=1)) * np.sqrt(N)
                t, p = np.unravel_index((sliced_arr / self.sig_thresh).argmax(), sliced_arr.shape)
                f = slice_dict[spw][shape]
                R = sliced_arr[t, p] / self.sig_thresh
            if R > 1:
                if R > R_max:
                    t_max, f_max, R_max = (t, f, R)
        return(t_max, f_max, R_max)

    def chisq_test(self, spw):

        p_min = 1
        shape_min = None
        for shape in self.slice_dict[spw]:
            if shape is 'point':
                p = 1
                f_point = None
                for f in range(self.INS.data.shape[2]):
                    stat, p_point = Util.chisq(self.INS.hist_make(sig_thresh=self.sig_thresh,
                                                                  event=[spw, slice(f, f + 1)]))
                    if p_point < p:
                        p = p_point
                        f_point = f
            else:
                stat, p = Util.chisq(self.INS.hist_make(sig_thresh=self.sig_thresh,
                                                        event=[spw, slice_dict[spw][shape]]))
            if p < p_min:
                p_min = p
                shape_min = shape

        return(p_min, shape_min)

    def apply_match_filter_test(self):

        count = 1
        while count:
            count = 0
            for spw in range(self.INS.data.shape[1]):
                t_max, f_max, R_max = self.match_test(spw)
                if R_max > -np.inf:
                    count += 1
                    self.INS.data[t_max, spw, f_max] = np.ma.masked
                    self.INS.events.append([spw, f_max])
                    self.INS.event_hists.append(self.INS.hist_make(sig_thresh=self.sig_thresh, event=[spw, f_max]))
            self.INS.data_ms = self.INS.mean_subtract()

    def apply_chisq_test(self):
        p_min = 0
        while p_min < 1:
            for spw in range(self.INS.data.shape[1]):
                p_min, shape_min = chisq_test(self, spw)
                if p_min < self.alpha:
                    if shape_min is 'point':
                        self.INS.chisq_hists.append(self.INS.hist_make(sig_thresh=self.sig_thresh,
                                                                       event=[spw, slice(f, f + 1)]))
                        self.INS.data[:, spw, slice(f_point, f_point + 1)] = np.ma.masked
                        self.INS.chisq_events.append([spw, slice(f_point, f_point + 1)])
                    else:
                        self.INS.chisq_hists.append(self.INS.hist_make(sig_thresh=self.sig_thresh,
                                                                       event=[spw, slice_dict[spw][shape]]))
                        self.INS.data[:, spw, slice_dict[spw][shape_min]] = np.ma.masked
                        self.INS.chisq_events.append([spw, slice_dict[spw][shape]])

            self.INS.data_ms = self.mean_subtract()

    def apply_samp_thresh_test(self):

        self.INS.data[:, np.count_nonzero(np.logical_not(self.INS.data.mask), axis=0) < self.N_thresh] = np.ma.masked
        self.INS.data_ms = self.mean_subtract()
