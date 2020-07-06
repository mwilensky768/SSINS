"""
Match Filter class
"""

import numpy as np


class MF():

    """
    Defines the Match Filter (MF) class.
    """

    def __init__(self, freq_array, sig_thresh, shape_dict={}, N_samp_thresh=0,
                 broadcast_dict=None, narrow=True, streak=True):

        """
        Args:
            freq_array: Sets the freq_array attribute of the filter
            sig_thresh (dict or number): If dictionary, the keys are the desired
                shapes to flag. The values are the desired significance
                thresholds for each shape. If streak or narrow are True, thresholds
                for these must be included in this dictionary, although they
                should not be included in the shape_dict keyword input. If passing
                a number, this number is used as the threshold for all shapes.
            shape_dict (dict): A dictionary of shapes to flag. Keys are shapes
                other than 'streak' and 'narrow'. Values are frequency limits
                of corresponding shape.
            N_samp_thresh (int): Sets the N_samp_thresh attribute of the filter
            broadcast_dict (dict): Optional. Describes how to partition the band
                when broadcasting over frequencies. The keys should be the names
                of each subband to broadcast over, and the values should be the
                edges of each subband (i.e. two-element lists).
            narrow (bool): If True, search for narrowband (single channel) RFI
            streak (bool): If True, search for broad RFI streaks that occupy the entire observing band
        """
        if (not shape_dict) and (not narrow) and (not streak):
            raise ValueError("There are not shapes in the shape_dict and narrow/streak shapes are disabled. Check keywords.")

        self.freq_array = freq_array
        """A 1-d array of frequencies (in hz) for the filter to operate with"""
        self.shape_dict = shape_dict
        """A dictionary of shapes. Keys are a shape name, values are the lower and upper frequency bounds in Hz."""
        self.sig_thresh = sig_thresh
        """A dictionary of significance thresholds to flag per shape. Keys are shapes and values are thresholds."""
        self.N_samp_thresh = N_samp_thresh
        """The threshold for flagging an entire channel when some flags exist and apply_samp_thresh() is called.
           See apply_samp_thresh() documentation for exact meaning."""
        self.slice_dict = self._shape_slicer(narrow, streak, shape_dict)
        """A dictionary whose keys are the same as shape_dict and whose values are corresponding slices into the freq_array attribute"""
        if type(self.sig_thresh) is dict:
            for shape in self.slice_dict:
                if shape not in self.sig_thresh.keys():
                    raise KeyError("%s shape has no sig_thresh. Check sig_thresh input." % shape)
        else:
            sig_thresh_dict = {}
            for shape in self.slice_dict:
                sig_thresh_dict[shape] = self.sig_thresh
            self.sig_thresh = sig_thresh_dict
        self.broadcast_dict = broadcast_dict
        """A dictionary of subbands. Keys are a subband name, values are the lower and upper frequency bounds in Hz."""
        if self.broadcast_dict is not None:
            self.broadcast_slc_dict = self._shape_slicer(False, False, input_dict="broadcast_dict")
            """A dictionary whose keys are the same as broadcast_dict and whose values are corresponding slices into the freq_array attribute"""

    def _shape_slicer(self, narrow, streak, input_dict="shape_dict"):

        """
        This function converts the frequency information in the shape_dict
        attribute to slice objects for the channel numbers of the spectrum.
        The narrow and streak shapes require special slices.

        Args:
            narrow (bool): If True, add the narrow shape to the dictionary
            streak (bool): If True, add the streak shape to the dictionary
            input_dict (str): Which dict attribute to operate on.

        Returns:
            slice_dict: See slice_dict attribute
        """

        slice_dict = {}
        for shape in getattr(self, input_dict):
            if min(self.freq_array) < min(getattr(self, input_dict)[shape]) or \
               max(self.freq_array) > max(getattr(self, input_dict)[shape]):
                min_chan = np.argmin(np.abs(self.freq_array - min(getattr(self, input_dict)[shape])))
                max_chan = np.argmin(np.abs(self.freq_array - max(getattr(self, input_dict)[shape])))
                # May have to extend the edges depending on if the shape extends beyond the min and max chan infinitesimally
                if (self.freq_array[min_chan] - min(getattr(self, input_dict)[shape]) > 0) and (min_chan > 0):
                    min_chan -= 1
                if self.freq_array[max_chan] - max(getattr(self, input_dict)[shape]) < 0:
                    max_chan += 1
                slice_dict[shape] = slice(min_chan, max_chan)
        if narrow:
            slice_dict['narrow'] = None
        if streak:
            slice_dict['streak'] = slice(0, len(self.freq_array))

        return(slice_dict)

    def match_test(self, INS):

        """
        The primary test that the filter is used with. The greatest outlier in
        each shape is put forth and a champion among them is chosen. The time,
        frequencies, and outlier statistic and shape of this champion is
        returned to the stack.

        Args:
            INS: An INS to test

        Returns:
            t_max: The time index of the strongest outlier (None if no significant outliers)
            f_max: The slice in the freq_array for the strongest outlier (None if no significant outliers)
            R_max: The ratio of the z-score of the outlier to the sig_thresh (-np.inf if no significant outliers)
            shape_max: The shape of the strongest outlier
        """

        sig_max = -np.inf
        t_max = None
        f_max = None
        shape_max = None
        for shape in self.slice_dict:
            if shape is 'narrow':
                t, f, p = np.unravel_index(np.absolute(INS.metric_ms).argmax(),
                                           INS.metric_ms.shape)
                sig = np.absolute(INS.metric_ms[t, f, p])
                f = slice(f, f + 1)
            else:
                N = np.count_nonzero(np.logical_not(INS.metric_ms[:, self.slice_dict[shape]].mask),
                                     axis=1)
                sliced_arr = np.absolute(INS.metric_ms[:, self.slice_dict[shape]].mean(axis=1)) * np.sqrt(N)
                t, p = np.unravel_index((sliced_arr / self.sig_thresh[shape]).argmax(),
                                        sliced_arr.shape)
                f = self.slice_dict[shape]
                sig = sliced_arr[t, p]
            if sig > self.sig_thresh[shape]:
                if sig > sig_max:
                    t_max, f_max, sig_max, shape_max = (t, f, sig, shape)
        return(t_max, f_max, sig_max, shape_max)

    def apply_match_test(self, INS, event_record=True, apply_samp_thresh=False,
                         freq_broadcast=False):

        """
        A method that uses the match_test() method to flag RFI. The champion
        from match_test() is flagged and the mean-subtracted spectrum is
        recalculated. This repeats until there are no more outliers greater than sig_thresh.
        Also can apply the samp_thresh_test in each iteration, which flags
        highly occupied channels between match test iterations.

        Args:
            INS: The INS to flag
            event_record (bool): If True, append events to INS.match_events
            apply_samp_thresh (bool): If True, call apply_samp_thresh() between iterations. Note this will not execute if the N_samp_thresh parameter is 0.
            freq_broadcast (bool): If True, broadcast flags between iterations using the broadcast_dict
        """

        count = 1
        while count:
            count = 0
            t_max, f_max, sig_max, shape_max = self.match_test(INS)
            if sig_max > -np.inf:
                count += 1
                event = (t_max, f_max, shape_max, sig_max)
                INS.metric_array[event[:2]] = np.ma.masked
                # Only adjust those values in the sig_array that are not already assigned
                nonmask = np.logical_not(INS.metric_ms.mask[event[:2]])
                INS.sig_array[event[:2]][nonmask] = INS.metric_ms[event[:2]][nonmask]
                if event_record:
                    INS.match_events.append(event)
                if freq_broadcast:
                    self.freq_broadcast(INS, event_record=event_record)
                if (apply_samp_thresh and self.N_samp_thresh):
                    self.apply_samp_thresh_test(INS, event_record=event_record)
                if not np.all(INS.metric_array[:, f_max, 0].mask):
                    INS.metric_ms[:, f_max] = INS.mean_subtract(freq_slice=f_max)
                else:
                    INS.metric_ms[:, f_max] = np.ma.masked
        nonmask_all = np.logical_not(INS.metric_ms.mask)
        INS.sig_array[nonmask_all] = INS.metric_ms[nonmask_all]

    def apply_samp_thresh_test(self, INS, event_record=False):
        """
        The sample threshold test. This tests to see if any channels have fewer
        unflagged channels than the N_samp_thresh parameter. If so, the entire channel is
        flagged. A ValueError is raised if the N_samp_thresh parameter is greater
        than the number of times in the observation, due to the fact that this
        will always lead to flagging the entire observation.

        Args:
            INS: An INS to test
            event_record (bool): If true, append events to INS.match_events
        """

        if self.N_samp_thresh > INS.metric_array.shape[0]:
            raise ValueError("N_samp_thresh parameter is set higher than "
                             "the number of time samples. This will "
                             "always result in flagging the entire "
                             "observation. Aborting flagging.")
        good_chans = np.where(np.logical_not(np.all(INS.metric_array[:, :, 0].mask, axis=0)))[0]
        N_unflagged = INS.metric_array.shape[0] - np.count_nonzero(INS.metric_array.mask[:, good_chans, 0], axis=0)
        if np.any(N_unflagged < self.N_samp_thresh):
            good_chan_ind = np.where(N_unflagged < self.N_samp_thresh)[0]
            if event_record:
                for chan in good_chans[good_chan_ind]:
                    event_times = np.nonzero(np.logical_not(INS.metric_array.mask[:, chan]))[0]
                    INS.sig_array[event_times, chan] = INS.metric_ms[event_times, chan]
                    for event_time in event_times:
                        event = (event_time, slice(chan, chan + 1), 'samp_thresh',
                                 None)
                        INS.match_events.append(event)
            INS.metric_array[:, good_chans[good_chan_ind]] = np.ma.masked

    def freq_broadcast(self, INS, event_record=False):
        """
        Broadcast flags in frequency. An event will be recorded in the
        match_filter saying which integration/band was flagged.
        """
        if self.broadcast_dict is None:
            raise ValueError("MF object does not have a broadcast_dict, but is "
                             " being asked to broadcast flags. Check "
                             " initialization of MF object.")

        for sb_name in self.broadcast_slc_dict:
            # Find times that are not all flagged but have some flags in this slice
            time_inds = np.logical_xor(np.any(ins.metric_array.mask[:, self.broadcast_slc_dict[sb_name]]),
                                       np.all(ins.metric_array.mask[:, self.broadcast_slc_dict[sb_name]]),
                                       axis=[1, -1])
            if np.any(time_inds):
                ins.metric_array[time_inds, self.broadcast_slc_dict[sb_name]] = np.ma.masked
                if event_record:
                    for event_time in np.arange(INS.Ntimes)[time_inds]:
                        event = (event_time, self.broadvast_slc_dict[sb_name],
                                 f'freq_broadcast_{sb_name}', None)
