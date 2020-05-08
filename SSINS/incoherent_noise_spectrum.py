"""
The incoherent noise spectrum class.
"""

import numpy as np
import os
from pyuvdata import UVFlag
import yaml
from SSINS import version
from functools import reduce
import warnings


class INS(UVFlag):
    """
    Defines the incoherent noise spectrum (INS) class, which is a subclass of
    the UVFlag class, a member of the pyuvdata software package.
    """

    def __init__(self, input, history='', label='', order=0, mask_file=None,
                 match_events_file=None, spectrum_type="cross"):

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

        super().__init__(input, mode='metric', copy_flags=False,
                         waterfall=False, history='', label='')
        if self.type == 'baseline':

            self.spectrum_type = spectrum_type
            """Type of visibilities used in spectrum. Either cross or auto. No mixing allowed."""
            self.history += f"Initialized spectrum_type:{self.spectrum_type} from visibility data. "

            if self.spectrum_type == "cross":

                has_autos = np.any(input.ant_1_array == input.ant_2_array)
                if has_autos:
                    warnings.warn("Requested spectrum type is 'cross'. Removing autos before averaging.")
                    input.select(ant_str="cross")

                has_crosses = np.any(input.ant_1_array != input.ant_2_array)
                if not has_crosses:
                    raise ValueError("Requested spectrum type is cross, but no cross"
                                     " correlations remain. Check SS input.")

            elif self.spectrum_type == "auto":
                has_crosses = np.any(input.ant_1_array != input.ant_2_array)
                if has_crosses:
                    warnings.warn("Requested specturm type is 'auto'. Removing"
                                  " crosses before averaging.")
                    input.select(ant_str="auto")

                has_autos = np.any(input.ant_1_array == input.ant_2_array)
                if not has_autos:
                    raise ValueError("Requested spectrum type is cross, but no autos"
                                     " remain. Check SS input.")

            self.metric_array = np.abs(input.data_array)
            """The baseline-averaged sky-subtracted visibility amplitudes (numpy masked array)"""
            self.weights_array = np.logical_not(input.data_array.mask)
            """The number of baselines that contributed to each element of the metric_array"""
            super().to_waterfall(method='mean')
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
        """An array containing the z-scores of the data in the incoherent noise spectrum."""
        self.sig_array = np.ma.copy(self.metric_ms)
        """An array that is initially equal to the z-score of each data point. During flagging,
        the entries are assigned according to their z-score at the time of their flagging."""

    def mean_subtract(self, freq_slice=slice(None), return_coeffs=False):

        """
        A function which calculated the mean-subtracted spectrum from the
        regular spectrum. A spectrum made from a perfectly clean observation
        will be written as a z-score by this operation.

        Args:
            freq_slice: The frequency slice over which to do the calculation. Usually not
               set by the user.
            return_coeffs: Whether or not to return the mean/polynomial coefficients

        Returns:
            MS (masked array): The mean-subtracted data array.
        """

        # This constant is determined by the Rayleigh distribution, which
        # describes the ratio of its rms to its mean
        C = 4 / np.pi - 1
        if not self.order:
            coeffs = self.metric_array[:, freq_slice].mean(axis=0)
            MS = (self.metric_array[:, freq_slice] / coeffs - 1) * np.sqrt(self.weights_array[:, freq_slice] / C)
        else:
            MS = np.zeros_like(self.metric_array[:, freq_slice])
            coeffs = np.zeros((self.order + 1, ) + MS.shape[1:])
            # Make sure x is not zero so that np.polyfit can proceed without nans
            x = np.arange(1, self.metric_array.shape[0] + 1)
            # We want to iterate over only a subset of the frequencies, so we need to investigate
            y_0 = self.metric_array[:, freq_slice, 0]
            # Find which channels are not fully masked (only want to iterate over those)
            # This gives an array of channel indexes into the freq_slice
            good_chans = np.where(np.logical_not(np.all(y_0.mask, axis=0)))[0]
            # Only do this if there are unmasked channels
            if len(good_chans) > 0:
                # Want to group things by unique mask for fastest implementation
                # mask_inv tells us which channels have the same mask (indexed into good_chans)
                unique_masks, mask_inv = np.unique(y_0[:, good_chans].mask, axis=1,
                                                   return_inverse=True)
                # np.ma.polyfit only takes 2d args, so have to iterate over pols
                for pol_ind in range(self.metric_array.shape[-1]):
                    good_data = self.metric_array[:, freq_slice, pol_ind][:, good_chans]
                    # Iterate over the unique masks grouping channels for fastest implementation
                    for mask_ind in range(unique_masks.shape[1]):
                        # Channels which share a mask (indexed into good_chans)
                        chans = np.where(mask_inv == mask_ind)[0]
                        y = good_data[:, chans]
                        coeff = np.ma.polyfit(x, y, self.order)
                        coeffs[:, good_chans[chans], pol_ind] = coeff
                        # Make the fit spectrum
                        mu = np.sum([np.outer(x**(self.order - poly_ind), coeff[poly_ind])
                                     for poly_ind in range(self.order + 1)],
                                    axis=0)
                        MS[:, good_chans[chans], pol_ind] = (y / mu - 1) * np.sqrt(self.weights_array[:, freq_slice, pol_ind][:, good_chans[chans]] / C)
            else:
                MS[:] = np.ma.masked

        if return_coeffs:
            return(MS, coeffs)
        else:
            return(MS)

    def mask_to_flags(self):
        """
        Propagates the mask to construct flags for the original
        (non time-differenced) data. If a time is flagged in the INS, then both
        times that could have contributed to that time in the sky-subtraction
        step are flagged in the new array.

        Returns:
            tp_flags (array): The time-propagated flags
        """

        # Propagate the flags
        shape = list(self.metric_array.shape)
        tp_flags = np.zeros([shape[0] + 1] + shape[1:], dtype=bool)
        tp_flags[:-1] = self.metric_array.mask
        tp_flags[1:] = np.logical_or(tp_flags[1:], tp_flags[:-1])

        return(tp_flags)

    def flag_uvf(self, uvf, inplace=False):
        """
        Applies flags calculated from mask_to_flags method onto a given UVFlag
        object. Option to edit an existing uvf object inplace. Works by
        propagating the mask on sky-subtracted data to flags that can be applied
        to the original data, pre-subtraction.  ORs the flags from the INS
        object and the input uvf object.

        Args:
            uvf: A waterfall UVFlag object in flag mode to apply flags to. Must be
                constructed from the original data. Errors if not waterfall,
                in flag mode, or time ordering does not match INS object.

            inplace: Whether to edit the uvf input inplace or not. Default False.

        Returns:
            uvf: The UVFlag object in flag mode with the time-propagated flags.
        """
        if uvf.mode != 'flag':
            raise ValueError("UVFlag object must be in flag mode to write flags from INS object.")
        if uvf.type != 'waterfall':
            raise ValueError("UVFlag object must be in waterfall mode to write flags from INS object.")
        test_times = 0.5 * (uvf.time_array[:-1] + uvf.time_array[1:])
        if not np.all(self.time_array == test_times):
            raise ValueError("UVFlag object's times do not match those of INS object.")

        new_flags = self.mask_to_flags()

        if inplace:
            this_uvf = uvf
        else:
            this_uvf = uvf.copy()

        this_uvf.flag_array = np.logical_or(this_uvf.flag_array, new_flags)

        return(this_uvf)

    def write(self, prefix, clobber=False, data_compression='lzf',
              output_type='data', mwaf_files=None, mwaf_method='add',
              Ncoarse=24, sep='_', uvf=None):

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
            sep (str): Determines the separator in the filename of the output file.
        """

        version_info_list = ['%s: %s, ' % (key, version.version_info[key]) for key in version.version_info]
        version_hist_substr = reduce(lambda x, y: x + y, version_info_list)
        if output_type == 'match_events':
            filename = '%s%sSSINS%s%s.yml' % (prefix, sep, sep, output_type)
        else:
            filename = '%s%sSSINS%s%s.h5' % (prefix, sep, sep, output_type)

        if output_type != 'mwaf':
            self.history += 'Wrote %s to %s using SSINS %s. ' % (output_type, filename, version_hist_substr)

        if output_type == 'data':
            self.metric_array = self.metric_array.data
            super().write(filename, clobber=clobber, data_compression=data_compression)
            self.metric_array = np.ma.masked_array(data=self.metric_array, mask=self.metric_ms.mask)

        elif output_type == 'z_score':
            z_uvf = self.copy()
            z_uvf.metric_array = np.copy(self.metric_ms.data)
            super(INS, z_uvf).write(filename, clobber=clobber, data_compression=data_compression)
            del z_uvf

        elif output_type == 'mask':
            mask_uvf = self.copy()
            mask_uvf.to_flag()
            mask_uvf.flag_array = np.copy(self.metric_array.mask)
            super(INS, mask_uvf).write(filename, clobber=clobber, data_compression=data_compression)
            del mask_uvf

        elif output_type == 'flags':
            if uvf is None:
                raise ValueError("When writing 'flags', you must supply a UVFlag"
                                 "object to write flags to using the uvf keyword.")
            flag_uvf = self.flag_uvf(uvf=uvf)
            flag_uvf.write(filename, clobber=clobber, data_compression=data_compression)

        elif output_type == 'match_events':
            yaml_dict = {'time_ind': [],
                         'freq_bounds': [],
                         'shape': [],
                         'sig': []}
            for event in self.match_events:
                yaml_dict['time_ind'].append(int(event[0]))
                # Convert slice object to just its bounds
                freq_bounds = [int(event[1].start), int(event[1].stop)]
                yaml_dict['freq_bounds'].append(freq_bounds)
                yaml_dict['shape'].append(event[2])
                if event[3] is not None:
                    yaml_dict['sig'].append(float(event[3]))
                else:
                    yaml_dict['sig'].append(event[3])
            with open(filename, 'w') as outfile:
                yaml.safe_dump(yaml_dict, outfile, default_flow_style=False)

        elif output_type == 'mwaf':
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
                    # Check that freq res and time res are compatible
                    freq_mod = NCHANS % (flags.shape[1] / Ncoarse)
                    time_mod = NSCANS % flags.shape[0]
                    assert freq_mod == 0, "Number of fine channels of mwaf input and INS are incompatible."
                    assert time_mod == 0, "Time axes of mwaf input and INS flags are incompatible."
                    freq_div = NCHANS / (flags.shape[1] / Ncoarse)
                    time_div = NSCANS / flags.shape[0]
                    Nant = mwaf_hdu[0].header['NANTENNA']
                    Nbls = Nant * (Nant + 1) // 2

                    # Repeat in time
                    time_rep_flags = np.repeat(flags, time_div, axis=0)
                    # Repeat in freq
                    freq_time_rep_flags = np.repeat(time_rep_flags, freq_div, axis=1)
                    # Repeat in bls
                    freq_time_bls_rep_flags = np.repeat(freq_time_rep_flags[:, np.newaxis, NCHANS * boxint: NCHANS * (boxint + 1)], Nbls, axis=1)
                    # This shape is on MWA wiki. Reshape to this shape.
                    new_flags = freq_time_bls_rep_flags.reshape((NSCANS * Nbls, NCHANS))
                    if mwaf_method == 'add':
                        mwaf_hdu[1].data['FLAGS'][new_flags] = 1
                    elif mwaf_method == 'replace':
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
            yaml_dict = yaml.safe_load(infile)

        match_events = []
        for i in range(len(yaml_dict['time_ind'])):
            # Convert bounds back to slice
            freq_slice = slice(yaml_dict['freq_bounds'][i][0],
                               yaml_dict['freq_bounds'][i][1])

            match_events.append((yaml_dict['time_ind'][i],
                                 freq_slice,
                                 yaml_dict['shape'][i],
                                 yaml_dict['sig'][i]))

        return(match_events)

    @property
    def _data_params(self):
        """Overrides UVFlag._data_params property to add additional datalike parameters to list"""

        # Prevents a bug that occurs during __init__
        if not hasattr(self, 'metric_ms'):
            return(None)
        else:
            UVFlag_params = super(INS, self)._data_params
            Extra_params = ['metric_ms', 'sig_array']
            SSINS_params = UVFlag_params + Extra_params
        return(SSINS_params)

    def select(self, **kwargs):
        """Thin wrapper around UVFlag.select that also recalculates the ms array
        immediately afterwards.
        """

        super(INS, self).select(**kwargs)
        self.metric_ms = self.mean_subtract()
