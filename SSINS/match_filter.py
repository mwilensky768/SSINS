"""
Match Filter class
"""

import numpy as np
import warnings
from collections import namedtuple
import yaml
from copy import deepcopy
from SSINS import version
from functools import reduce
import os

Event = namedtuple("Event", ["time_slice", "freq_slice", "shape", "sig"])


class MF():

    """
    Defines the Match Filter (MF) class.
    """

    def __init__(self, freq_array, sig_thresh, shape_dict={}, tb_aggro=0,
                 N_samp_thresh=None, narrow=True, streak=True, broadcast_dict={},
                 broadcast_streak=False):

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
            broadcast_streak (bool): Optional, broadcast flags over whole
                observing band.
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
        self.tb_aggro = tb_aggro
        """The threshold for flagging an entire channel when some flags exist
           and apply_samp_thresh() is called. Must be between 0 and 1. Represents
           a fraction of unflagged data remaining."""
        if (self.tb_aggro >= 1) or (self.tb_aggro < 0):
            raise ValueError("tb_aggro parameter must be between 0 and 1.")
        if N_samp_thresh is not None:
            raise ValueError("The N_samp_thresh parameter is now deprected. See"
                             " the tb_aggro parameter for its replacement.")
        self.slice_dict = self._shape_slicer(narrow, streak, "shape_dict")
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

        self.broadcast_slc_dict = self._shape_slicer(False, broadcast_streak,
                                                     input_dict="broadcast_dict")
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
        ch_wid = self.freq_array[1] - self.freq_array[0]
        for shape in getattr(self, input_dict):
            if min(self.freq_array) <= min(getattr(self, input_dict)[shape]) or \
               max(self.freq_array) >= max(getattr(self, input_dict)[shape]):
                # Assuming frequencies represent channel centers at that fine channel bandpass has sharp cutoff at midpoint
                min_chan = np.argmin(np.abs(self.freq_array - min(getattr(self, input_dict)[shape])))
                # Extend by 1 so that it is inclusive at the upper boundary
                max_chan = np.argmin(np.abs(self.freq_array - max(getattr(self, input_dict)[shape]))) + 1
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
            if shape == 'narrow':
                t, f, p = np.unravel_index(np.absolute(INS.metric_ms).argmax(),
                                           INS.metric_ms.shape)
                sig = np.absolute(INS.metric_ms[t, f, p])
                t = slice(t, t + 1)
                f = slice(f, f + 1)
            else:
                N = np.count_nonzero(np.logical_not(INS.metric_ms[:, self.slice_dict[shape]].mask),
                                     axis=1)
                sliced_arr = np.absolute(INS.metric_ms[:, self.slice_dict[shape]].mean(axis=1)) * np.sqrt(N)
                t, p = np.unravel_index((sliced_arr / self.sig_thresh[shape]).argmax(),
                                        sliced_arr.shape)
                t = slice(t, t + 1)
                f = self.slice_dict[shape]
                # Pull out the number instead of a sliced arr
                sig = sliced_arr[t, p][0]
            if sig > self.sig_thresh[shape]:
                if sig > sig_max:
                    t_max, f_max, shape_max, sig_max = (t, f, shape, sig)

        if shape_max == "narrow":
            shape_max = "narrow_%.3fMHz" % (INS.freq_array[f_max][0] * 10**(-6))

        event = Event(t_max, f_max, shape_max, sig_max)

        return(event)

    def apply_match_test(self, INS, event_record=True, apply_samp_thresh=None,
                         freq_broadcast=False, time_broadcast=False):

        """
        A method that uses the match_test() method to flag RFI. The champion
        from match_test() is flagged and the mean-subtracted spectrum is
        recalculated. This repeats until there are no more outliers greater than sig_thresh.
        Also can apply the samp_thresh_test in each iteration, which flags
        highly occupied channels between match test iterations.

        Args:
            INS: The INS to flag
            event_record (bool): If True, append events to INS.match_events
            apply_samp_thresh (bool): Deprecated in favor of the time_broadcast keyword.
            freq_broadcast (bool): If True, broadcast flags between iterations using the broadcast_dict
            time_broadcast (bool): If True, broadcasts flags in time if significant flagging in channels. Set tb_aggro parameter for aggression.
        """
        if apply_samp_thresh is not None:
            raise ValueError("apply_samp_thresh has been deprecated in favor of"
                             " the time_broadcast keyword.")

        # Initialize the counter so the loop starts.
        count = 1
        while count:
            # If no events are found, this will remain 0, and the loop will end
            count = 0
            event = self.match_test(INS)
            if event.sig > -np.inf:
                count += 1
                INS.metric_array[event[:2]] = np.ma.masked
                # Only adjust those values in the sig_array that are not already assigned
                nonmask = np.logical_not(INS.metric_ms.mask[event[:2]])
                INS.sig_array[event[:2]][nonmask] = INS.metric_ms[event[:2]][nonmask]
                if event_record:
                    INS.match_events.append(event)
                if time_broadcast:
                    event = self.time_broadcast(INS, event, event_record=event_record)
                if freq_broadcast:
                    event = self.freq_broadcast(INS, event, event_record=event_record)
                if not np.all(INS.metric_array[:, event[1]].mask):
                    INS.metric_ms[:, event[1]] = INS.mean_subtract(freq_slice=event[1])
                else:
                    INS.metric_ms[:, event[1]] = np.ma.masked
        nonmask_all = np.logical_not(INS.metric_ms.mask)
        INS.sig_array[nonmask_all] = INS.metric_ms[nonmask_all]

    def time_broadcast(self, INS, event, event_record=False):
        """
        Broadcasts flags in time for a subband (determined by the passed event)
        if the fraction of unflagged samples in the subband is less than the
        tb_aggro parameter. If so, the entire subband is flagged for the whole
        object. A ValueError is raised if the aggro parameter is greater
        than or equal to 1, which will always flag everything.

        Args:
            INS: An INS to test
            event: The event to check.
            event_record (bool): If true, append events to INS.match_events.
        Returns:
            new_event: Possible new event if more flagging happened. Returns old
                event if no additional flagging happened.
        """

        # Find how many channels are already fully flagged, so we can ignore them
        num_chans_all_flag = np.sum(np.all(INS.metric_array.mask[:, event[1], :], axis=(0, -1)))
        # Find the total data volume and subtract off the data volume in channels that are totally flagged
        total = np.prod(INS.metric_array[:, event[1]].shape)
        total_invalid = num_chans_all_flag * INS.Ntimes * INS.Npols
        total_valid = total - total_invalid
        # Find the total flagged data volume and subtract off the invalid data
        total_flag = np.sum(INS.metric_array[:, event[1]].mask)
        total_flag_valid = total_flag - total_invalid
        # Find the flag fraction, unflagged fraction, compare to aggro parameter
        flag_frac = total_flag_valid / total_valid
        unflag_frac = 1 - flag_frac
        if unflag_frac <= self.tb_aggro:
            INS.metric_array[:, event[1]] = np.ma.masked
            if event_record:
                new_event = Event(slice(0, INS.Ntimes), event[1],
                                  f"time_broadcast_{event[2]}", None)
                INS.match_events.append(new_event)
        else:
            new_event = event

        return(new_event)

    def freq_broadcast(self, INS, event, event_record=False):
        """
        Broadcast flags in frequency, regardless of flagging fraction. Determined,
        by the broadcast_dict. An event will be recorded in the
        match_filter saying which integration/band was flagged.

        Args:
            INS: The incoherent noise spectrum being worked on.
            event: The event to broadcast flags for.
            event_record (bool): Whether to record a new event for this flagging entry.
        """
        if self.broadcast_slc_dict == {}:
            raise ValueError("MF object does not have a broadcast_dict, but is "
                             " being asked to broadcast flags. Check "
                             " initialization of MF object.")

        new_event_set = set()
        sbs = []
        for sb in self.broadcast_slc_dict:
            sb_slc = self.broadcast_slc_dict[sb]
            event_set = set(np.arange(event[1].stop)[event[1]])
            broad_set = set(np.arange(sb_slc.stop)[sb_slc])
            if not event_set.isdisjoint(broad_set):
                new_event_set = new_event_set.union(broad_set)
                sbs.append(sb)
        if len(new_event_set) > 0:
            sb_string = "_"
            sb_string = sb_string.join(sbs)
            # They should all be contiguous until discontiguous shapes are allowed
            new_event_slc = slice(min(new_event_set), max(new_event_set) + 1)
            INS.metric_array[event[0], new_event_slc] = np.ma.masked
            final_event = Event(event[0], new_event_slc, f"freq_broadcast_{sb_string}", None)
            if event_record:
                INS.match_events.append(final_event)
        else:
            final_event = event

        return(final_event)

    def write(self, prefix, sep="_", clobber=False):
        """
        Writes out a yaml file with the important information about the filter.

        Args:
            prefix: The filepath prefix for the output file. Output file will be
                named f'{prefix}{sep}matchfilter.yaml'
            sep: The separator character between the prefix and the rest of the output filepath.
            clobber: Whether to overwrite an identically named file. True overwrites.
        """

        outpath = f"{prefix}{sep}SSINS{sep}matchfilter.yml"

        yaml_dict = self._make_yaml_dict()

        file_exists = os.path.exists(outpath)

        if file_exists and not clobber:
            raise ValueError(f"matchfilter file with prefix {prefix} already exists and clobber is False.")
        else:
            with open(outpath, 'w') as outfile:
                yaml.safe_dump(yaml_dict, outfile)

    def _make_yaml_dict(self):
        """
        Helper function for MF.write that sets up the dictionary for the yaml output.
        """

        broadcast_dict = deepcopy(self.broadcast_dict)
        # Include additional shape if in the slc_dict which may be missing from the broadcast_dict
        if "streak" in self.broadcast_slc_dict:
            broadcast_dict.update({"streak": [self.freq_array[0], self.freq_array[-1]]})

        shape_dict = deepcopy(self.shape_dict)
        if "streak" in self.slice_dict:
            shape_dict.update({"streak": [self.freq_array[0], self.freq_array[-1]]})
        if "narrow" in self.slice_dict:
            # Placeholder values. "narrow" really refers to Nfreqs different shapes.
            shape_dict.update({"narrow (vals are placeholders)": [self.freq_array[0], self.freq_array[-1]]})

        version_info_list = [f'%s: %s, ' % (key, version.version_info[key]) for key in version.version_info]
        version_hist_substr = reduce(lambda x, y: x + y, version_info_list)

        yaml_dict = {"freqs": [float(freq) for freq in self.freq_array],
                     "shape_dict": {shape: [float(shape_dict[shape][0]), float(shape_dict[shape][1])] for shape in shape_dict},
                     "sig_thresh": self.sig_thresh,
                     "tb_aggro": self.tb_aggro,
                     "freq_broadcast": {shape: [float(broadcast_dict[shape][0]), float(broadcast_dict[shape][1])] for shape in broadcast_dict},
                     "version_info": version_hist_substr}

        return(yaml_dict)
