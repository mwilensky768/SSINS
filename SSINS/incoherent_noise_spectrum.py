from __future__ import absolute_import, division, print_function

"""
The incoherent noise spectrum class.
"""

import numpy as np
from scipy.special import erfcinv
import os
import warnings
import pickle
from pyuvdata import UVFlag
import yaml
from SSINS import version
from functools import reduce


class INS(UVFlag):
    """
    Defines the incoherent noise spectrum (INS) class, which is a subclass of
    the UVFlag class, a member of the pyuvdata software package.
    """

    def __init__(self, input, history='', label='', order=0, mask_file=None,
                 match_events_file=None):

        """
        init function for the INS class.

        Args:
            input: See UVFlag documentation
            history: See UVFlag documentation
            label: See UVFlag documentation
            order: Sets the order parameter for the INS object
            mask_file: A path to an .h5 (UVFlag) file that contains a mask for the metric_array
            match_events_file: A path to a .yml file that has events caught by the match filter
        """

        super(INS, self).__init__(input, mode='metric', copy_flags=False,
                                  waterfall=False, history='', label='')
        if self.type is 'baseline':
            # Set the metric array to the data array without the spw axis
            self.metric_array = np.abs(input.data_array)
            """The baseline-averaged sky-subtracted visibility amplitudes (numpy masked array)"""
            self.weights_array = np.logical_not(input.data_array.mask)
            """The number of baselines that contributed to each element of the metric_array"""
            super(INS, self).to_waterfall(method='mean')
        if not hasattr(self.metric_array, 'mask'):
            self.metric_array = np.ma.masked_array(self.metric_array)
        if mask_file is None:
            # Only mask elements initially if no baselines contributed
            self.metric_array.mask = self.weights_array == 0
        else:
            # Read in the flag array
            flag_uvf = UVFlag(mask_file)
            self.metric_array.mask = np.copy(flag_uvf.flag_array)
            del flag_uvf

        if match_events_file is None:
            self.match_events = []
            """A list of tuples that contain information about events caught during match filtering"""
        else:
            self.match_events = self.match_events_read(match_events_file)

        self.order = order
        """The order of polynomial fit for each frequency channel during mean-subtraction. Default is 0, which just calculates the mean."""
        self.metric_ms = self.mean_subtract()
        """The incoherent noise spectrum, after mean-subtraction."""

    def mean_subtract(self, f=slice(None)):

        """
        A function which calculated the mean-subtracted spectrum from the
        regular spectrum. A spectrum made from a perfectly clean observation
        will be written as a z-score by this operation.

        Args:
            f: The frequency slice over which to do the calculation. Usually not
               set by the user.

        Returns:
            MS (masked array): The mean-subtracted data array.
        """

        # This constant is determined by the Rayleigh distribution, which
        # describes the ratio of its rms to its mean
        C = 4 / np.pi - 1
        if not self.order:
            MS = (self.metric_array[:, f] / self.metric_array[:, f].mean(axis=0) - 1) * np.sqrt(self.weights_array[:, f] / C)
        else:
            MS = np.zeros_like(self.metric_array[:, f])
            # Make sure x is not zero so that np.polyfit can proceed without nans
            x = np.arange(1, self.metric_array.shape[0] + 1)
            for i in range(self.metric_array.shape[-1]):
                y = self.metric_array[:, f, i]
                # Only iterate over channels that are not fully masked
                good_chans = np.where(np.logical_not(np.all(y.mask, axis=0)))[0]
                # Create a subarray mask of only those good channels
                good_data = y[:, good_chans]
                # Find the set of unique masks so that we can iterate over only those
                unique_masks, mask_inv = np.unique(good_data.mask, axis=1,
                                                   return_inverse=True)
                for k in range(unique_masks.shape[1]):
                    # Channels which share a mask
                    chans = np.where(mask_inv == k)[0]
                    coeff = np.ma.polyfit(x, good_data[:, chans], self.order)
                    mu = np.sum([np.outer(x**(self.order - k), coeff[k]) for k in range(self.order + 1)],
                                axis=0)
                    MS[:, good_chans[chans], i] = (good_data[:, chans] / mu - 1) * np.sqrt(self.weights_array[:, f, i][:, good_chans[chans]] / C)

        return(MS)

    def mask_to_flags(self):
        """
        A function that propagates a mask on sky-subtracted data to flags that
        can be applied to the original data, pre-subtraction. The flags are
        propagated in such a way that if a time is flagged in the INS, then
        both times that could have contributed to that time in the sky-subtraction
        step are flagged.

        Returns:
            flags: The final flag array obtained from the mask.
        """
        shape = list(self.metric_array.shape)
        flags = np.zeros([shape[0] + 1] + shape[1:], dtype=bool)
        flags[:-1] = self.metric_array.mask
        flags[1:] = np.logical_or(flags[1:], flags[:-1])

        return(flags)

    def write(self, prefix, clobber=False, data_compression='lzf',
              output_type='data', mwaf_files=None, mwaf_method='add'):

        """
        Writes attributes specified by output_type argument to appropriate files
        with a prefix given by prefix argument. Can write mwaf files if required
        mwaf keywords arguments are provided. Required mwaf keywords are not
        required for any other purpose.

        Args:
            prefix: The filepath prefix for the output file e.g. /analysis/SSINS_outdir/obsid
            clobber: See UVFlag documentation
            data_compression: See UVFlag documentation
            output_type ('data', 'z_score', 'mask', 'flags', 'match_events'):

                data - outputs the metric_array attribute into an h5 file

                z_score - outputs the the metric_ms attribute into an h5 file

                mask - outputs the mask for the metric_array attribute into an h5 file

                flags - converts mask to flag using mask_to_flag() method and writes to an h5 file readable by UVFlag

                match_events - Writes the match_events attribute out to a human-readable yml file

                mwaf - Writes an mwaf file by converting mask to flags.
            mwaf_files (seq): A list of paths to mwaf files to use as input for each coarse channel
            mwaf_method ('add' or 'replace'): Choose whether to add SSINS flags to current flags in input file or replace them entirely
        """

        version_info_list = ['%s: %s, ' % (key, version.version_info[key]) for key in version.version_info]
        version_hist_substr = reduce(lambda x, y: x + y, version_info_list)
        if output_type is 'match_events':
            filename = '%s_SSINS_%s.yml' % (prefix, output_type)
        else:
            filename = '%s_SSINS_%s.h5' % (prefix, output_type)

        if output_type is not 'mwaf':
            self.history += 'Wrote %s to %s using SSINS %s. ' % (output_type, filename, version_hist_substr)

        if output_type is 'data':
            self.metric_array = self.metric_array.data
            super(INS, self).write(filename, clobber=clobber, data_compression=data_compression)
            self.metric_array = np.ma.masked_array(data=self.metric_array, mask=self.metric_ms.mask)

        elif output_type is 'z_score':
            z_uvf = self.copy()
            z_uvf.metric_array = np.copy(self.metric_ms.data)
            super(INS, z_uvf).write(filename, clobber=clobber, data_compression=data_compression)
            del z_uvf

        elif output_type is 'mask':
            mask_uvf = self.copy()
            mask_uvf.to_flag()
            mask_uvf.flag_array = np.copy(self.metric_array.mask)
            super(INS, mask_uvf).write(filename, clobber=clobber, data_compression=data_compression)
            del mask_uvf

        elif output_type is 'flags':
            flag_uvf = self.copy()
            flag_uvf.to_flag()
            flag_uvf.flag_array = self.mask_to_flags()
            super(INS, flag_uvf).write(filename, clobber=clobber, data_compression=data_compression)
            del flag_uvf

        elif output_type is 'match_events':
            yaml_dict = {'time_ind': [],
                         'freq_slice': [],
                         'shape': [],
                         'sig': []}
            for event in self.match_events:
                yaml_dict['time_ind'].append(event[0])
                yaml_dict['freq_slice'].append(event[1])
                yaml_dict['shape'].append(event[2])
                yaml_dict['sig'].append(event[3])
            with open(filename, 'w') as outfile:
                yaml.dump(yaml_dict, outfile, default_flow_style=False)

        elif output_type is 'mwaf':
            if mwaf_files is None:
                raise ValueError("mwaf_files is set to None. This must be a sequence of existing mwaf filepaths.")

            from astropy.io import fits
            flags = self.mask_to_flags()[:, :, 0]
            for path in mwaf_files:
                if not os.path.exists(path):
                    raise IOError("filepath %s in mwaf_files was not found in system." % path)
                path_ind = path.rfind('_') + 1
                boxstr = path[path_ind:path_ind + 2]
                boxint = int(boxstr) - 1
                with fits.open(path) as mwaf_hdu:
                    NCHANS = mwaf_hdu[0].header['NCHANS']
                    NSCANS = mwaf_hdu[0].header['NSCANS']
                    # 24 is the number of coarse channels in MWA data
                    assert NCHANS == (flags.shape[1] / 24), "Number of fine channels of mwaf input and INS do not match."
                    assert NSCANS == flags.shape[0], "Time axes of mwaf input and INS flags do not match"
                    Nant = mwaf_hdu[0].header['NANTENNA']
                    Nbls = Nant * (Nant + 1) // 2

                    # This shape is on MWA wiki
                    new_flags = np.repeat(flags[:, np.newaxis, NCHANS * boxint: NCHANS * (boxint + 1)], Nbls, axis=1).reshape((NSCANS * Nbls, NCHANS))
                    if mwaf_method is 'add':
                        mwaf_hdu[1].data['FLAGS'][new_flags] = 1
                    elif mwaf_method is 'replace':
                        mwaf_hdu[1].data['FLAGS'] = new_flags
                    else:
                        raise ValueError("mwaf_method is %s. Options are 'add' or 'replace'." % mwaf_method)

                    mwaf_hdu[0].header['SSINSVER'] = version_hist_substr

                    filename = '%s_%s.mwaf' % (prefix, boxstr)

                    mwaf_hdu.writeto(filename, overwrite=clobber)
                    self.history += 'Wrote flags to %s using SSINS %s' % (filename, version_hist_substr)
        else:
            raise ValueError("output_type %s is invalid. See documentation for options." % output_type)

    def match_events_read(self, filename):
        """
        Reads match events from file specified by filename argument

        Args:
            filename: The yml file with the stored match_events

        Returns:
            match_events: The match_events in the yml file
        """

        with open(filename, 'r') as infile:
            yaml_dict = yaml.load(infile)

        match_events = []
        for i in range(len(yaml_dict['time_ind'])):
            match_events.append((yaml_dict['time_ind'][i],
                                 yaml_dict['freq_slice'][i],
                                 yaml_dict['shape'][i],
                                 yaml_dict['sig'][i]))

        return(match_events)
