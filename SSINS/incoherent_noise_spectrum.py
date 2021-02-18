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
from itertools import combinations
from SSINS.match_filter import Event


class INS(UVFlag):
    """
    Defines the incoherent noise spectrum (INS) class, which is a subclass of
    the UVFlag class, a member of the pyuvdata software package.
    """

    def __init__(self, input, history='', label='', order=0, mask_file=None,
                 match_events_file=None, spectrum_type="cross",
                 use_integration_weights=False, nsample_default=1):

        """
        init function for the INS class.

        Args:
            input: See UVFlag documentation
            history: See UVFlag documentation
            label: See UVFlag documentation
            order: Sets the order parameter for the INS object
            mask_file: A path to an .h5 (UVFlag) file that contains a mask for the metric_array
            match_events_file: A path to a .yml file that has events caught by the match filter
            spectrum_type: Type of visibilities to use in making the specturm. Options are 'auto' or 'cross'.
            use_integration_weights: Whether to use the integration time and nsample array to compute the weights
            nsample_default: The default nsample value to fill zeros in the
                nsample_array with when there are some nsample=0. Important when
                working with data from uvfits files, which combine information
                from the flag_array and nsample_array in the weights field of
                the uvfits file.
        """

        super().__init__(input, mode='metric', copy_flags=False,
                         waterfall=False, history='', label='')

        # Used in _data_params to determine when not to return None
        self._super_complete = True

        if np.any(self.polarization_array > 0):
            raise ValueError("SS input has pseudo-Stokes data. SSINS does not"
                             " currently support pseudo-Stokes spectra.")

        self.spectrum_type = spectrum_type
        """The type of visibilities the spectrum was made from."""
        if self.spectrum_type not in ['cross', 'auto']:
            raise ValueError("Requested spectrum_type is invalid. Choose 'cross' or 'auto'.")

        spec_type_str = f"Initialized spectrum_type:{self.spectrum_type} from visibility data. "

        self.order = order
        """The order of polynomial fit for each frequency channel during mean-subtraction. Default is 0, which just calculates the mean."""

        if self.type == 'baseline':

            self.history += spec_type_str
            # Check if the data has a mask yet. If not, mask it and set flag_choice to None.
            if not isinstance(input.data_array, np.ma.MaskedArray):
                input.apply_flags()

            self.metric_array = np.abs(input.data_array)
            """The baseline-averaged sky-subtracted visibility amplitudes (numpy masked array)"""
            self.weights_array = np.logical_not(input.data_array.mask).astype(float)
            """The number of baselines that contributed to each element of the metric_array"""
            if use_integration_weights:
                # Set nsample default if some are zero
                input.nsample_array[input.nsample_array == 0] = nsample_default
                # broadcast problems with single pol
                self.weights_array *= (input.integration_time[:, np.newaxis, np.newaxis, np.newaxis] * input.nsample_array)

            cross_bool = self.ant_1_array != self.ant_2_array
            auto_bool = self.ant_1_array == self.ant_2_array

            if self.spectrum_type == "cross":

                has_crosses = np.any(cross_bool)
                if not has_crosses:
                    raise ValueError("Requested spectrum type is 'cross', but no cross"
                                     " correlations exist. Check SS input.")

                has_autos = np.any(auto_bool)
                if has_autos:
                    warnings.warn("Requested spectrum type is 'cross'. Removing autos before averaging.")
                    self.select(ant_str="cross")

            elif self.spectrum_type == "auto":
                has_autos = np.any(auto_bool)
                if not has_autos:
                    raise ValueError("Requested spectrum type is 'auto', but no autos"
                                     " exist. Check SS input.")

                has_crosses = np.any(cross_bool)
                if has_crosses:
                    warnings.warn("Requested spectrum type is 'auto'. Removing"
                                  " crosses before averaging.")
                    self.select(ant_str="auto")

            super().to_waterfall(method='mean', return_weights_square=True)
        # Make sure the right type of spectrum is being used, otherwise raise errors.
        # If neither statement inside is true, then it is an old spectrum and is therefore a cross-only spectrum.
        elif spec_type_str not in self.history:
            if "Initialized spectrum_type:" in self.history:
                raise ValueError("Requested spectrum type disagrees with saved spectrum. "
                                 "Make opposite choice on initialization.")
            elif self.spectrum_type == "auto":
                raise ValueError("Reading in a 'cross' spectrum as 'auto'. Check"
                                 " spectrum_type for INS initialization.")
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

        # For backwards compatibilty before weights_square_array was a thing
        # Works because weights are all 1 or 0 before this feature was added
        if self.weights_square_array is None:
            self.weights_square_array = np.copy(self.weights_array)
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

        if self.spectrum_type == 'cross':
            # This constant is determined by the Rayleigh distribution, which
            # describes the ratio of its rms to its mean
            C = 4 / np.pi - 1
        else:
            # This involves another constant that results from the folded normal distribution
            # which describes the amplitudes of the auto-pols.
            # The cross-pols have Rayleigh distributed amplitudes.
            C_ray = 4 / np.pi - 1
            C_fold = np.pi / 2 - 1
            C_pol_map = {-1: C_fold, -2: C_fold, -3: C_ray, -4: C_ray,
                         -5: C_fold, -6: C_fold, -7: C_ray, -8: C_ray}

            C = np.array([C_pol_map[pol] for pol in self.polarization_array])

        if not self.order:
            coeffs = np.ma.average(self.metric_array[:, freq_slice], axis=0, weights=self.weights_array[:, freq_slice])
            weights_factor = self.weights_array[:, freq_slice] / np.sqrt(C * self.weights_square_array[:, freq_slice])
            MS = (self.metric_array[:, freq_slice] / coeffs - 1) * weights_factor
        else:
            MS = np.zeros_like(self.metric_array[:, freq_slice])
            coeffs = np.zeros((self.order + 1, ) + MS.shape[1:])
            # Make sure x is not zero so that np.polyfit can proceed without nans
            x = np.arange(1, self.metric_array.shape[0] + 1)
            # We want to iterate over only a subset of the frequencies, so we need to investigate
            y_0 = self.metric_array[:, freq_slice]
            # Find which channels are not fully masked (only want to iterate over those)
            # This gives an array of channel indexes into the freq_slice
            good_chans = np.where(np.logical_not(np.all(y_0.mask, axis=0)))[0]
            # Only do this if there are unmasked channels
            if len(good_chans) > 0:
                # np.ma.polyfit does not take 2-d weights (!!!) so we just do the slow implementation and go chan by chan, pol-by-pol
                for chan in good_chans:
                    for pol_ind in range(self.Npols):
                        y = self.metric_array[:, chan, pol_ind]
                        w = self.weights_array[:, chan, pol_ind]
                        w_sq = self.weights_square_array[:, chan, pol_ind]
                        # Make the fit
                        coeff = np.ma.polyfit(x, y, self.order, w=w)
                        coeffs[:, chan, pol_ind] = coeff
                        # Do the magic
                        mu = np.sum([coeff[poly_ind] * x**(self.order - poly_ind)
                                     for poly_ind in range(self.order + 1)],
                                    axis=0)
                        weights_factor = w / np.sqrt(C * w_sq)
                        MS[:, chan, pol_ind] = (y / mu - 1) * weights_factor
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
        try:
            test_times = 0.5 * (uvf.time_array[:-1] + uvf.time_array[1:])
            time_compat = np.all(self.time_array == test_times)
            assert time_compat
        except Exception:
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
              metafits_file=None, Ncoarse=24, sep='_', uvf=None):

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
            mwaf_files (seq): A list of paths to mwaf files to use as input for
                each coarse channel
            mwaf_method ('add' or 'replace'): Choose whether to add SSINS flags
                to current flags in input file or replace them entirely
            metafits_file (str): A path to the metafits file if writing mwaf outputs.
                Required only if writing mwaf files.
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
            mask_uvf = self._make_mask_copy()
            super(INS, mask_uvf).write(filename, clobber=clobber, data_compression=data_compression)
            del mask_uvf

        elif output_type == 'flags':
            if uvf is None:
                raise ValueError("When writing 'flags', you must supply a UVFlag"
                                 "object to write flags to using the uvf keyword.")
            flag_uvf = self.flag_uvf(uvf=uvf)
            flag_uvf.write(filename, clobber=clobber, data_compression=data_compression)

        elif output_type == 'match_events':
            yaml_dict = {'time_bounds': [],
                         'freq_bounds': [],
                         'shape': [],
                         'sig': []}
            for event in self.match_events:
                time_bounds = [int(event[0].start), int(event[0].stop)]
                yaml_dict['time_bounds'].append(time_bounds)
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
            if metafits_file is None:
                raise ValueError("If writing mwaf files, must supply corresponding metafits file.")

            from astropy.io import fits
            flags = self.mask_to_flags()[:, :, 0]
            with fits.open(metafits_file) as meta_hdu_list:
                coarse_chans = meta_hdu_list[0].header["CHANNELS"].split(",")
                coarse_chans = np.sort([int(chan) for chan in coarse_chans])
            # Coarse channels need to be mapped properly
            # Up to coarse channel 128, the channels go in the right order
            # Then they go in reverse order
            # The number of properly ordered channels
            num_less = np.count_nonzero(coarse_chans <= 128)
            # Numbers associated with filenames
            box_keys = [str(ind).zfill(2) for ind in range(1, len(coarse_chans) + 1)]
            # Channel index associated with each box
            box_vals = np.zeros(len(coarse_chans), dtype=int)
            # The first num_less go in frequency-increasing order
            box_vals[:num_less] = np.arange(num_less)
            # The rest go in frequency-decreasing order
            box_vals[num_less:] = np.arange(len(coarse_chans) - 1, num_less - 1, -1)
            box_label_to_chan_ind_map = dict(zip(box_keys, box_vals))
            for path in mwaf_files:
                if not os.path.exists(path):
                    raise IOError("filepath %s in mwaf_files was not found in system." % path)
                path_ind = path.rfind('_') + 1
                boxstr = path[path_ind:path_ind + 2]
                chan_ind = box_label_to_chan_ind_map[boxstr]
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
                    freq_time_bls_rep_flags = np.repeat(freq_time_rep_flags[:, np.newaxis, NCHANS * chan_ind: NCHANS * (chan_ind + 1)], Nbls, axis=1)
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
        for i in range(len(yaml_dict['sig'])):
            # Convert bounds back to slice
            time_slice = slice(*yaml_dict['time_bounds'][i])
            freq_slice = slice(*yaml_dict['freq_bounds'][i])

            match_events.append(Event(time_slice,
                                      freq_slice,
                                      yaml_dict['shape'][i],
                                      yaml_dict['sig'][i]))

        return(match_events)

    @property
    def _data_params(self):
        """Overrides UVFlag._data_params property to add additional datalike parameters to list"""

        # Prevents a bug that occurs during __init__
        if not hasattr(self, '_super_complete'):
            return(None)
        else:
            UVFlag_params = super(INS, self)._data_params
            Extra_params = ['metric_ms', 'sig_array']
            SSINS_params = UVFlag_params + Extra_params
        return(SSINS_params)

    def select(self, inplace=True, **kwargs):
        """Thin wrapper around UVFlag.select that also recalculates the ms array
        immediately afterwards.

        args:
            inplace: Whether to do the operation inplace or return a copy without
                touching the original.

        returns:
            ins: INS object. Only returned if inplace is False.
        """
        if inplace:
            ins = self
        else:
            ins = self.copy()

        mask_uvf = ins._make_mask_copy()
        super(INS, ins).select(inplace=True, **kwargs)
        super(INS, mask_uvf).select(inplace=True, **kwargs)

        ins.metric_array.mask = np.copy(mask_uvf.flag_array)
        # In case this is called in the middle of the constructor.
        if hasattr(ins, 'metric_ms'):
            ins.metric_ms = ins.mean_subtract()

        if not inplace:
            return(ins)

    def __add__(self, other, inplace=False, axis="time", run_check=True,
                check_extra=True, run_check_acceptability=True):
        """
        Wrapper around UVFlag.__add__ that keeps track of the masks on the data.
            Args:
                other: Another INS object to add
                inplace: Whether to do the addition inplace or return a new INS
                axis: The axis over which to concatenate the objects
                run_check: Option to check for the existence and proper shapes
                    of parameters after combining two objects.
                check_extra: Option to check optional parameters as well as required ones.
                run_check_acceptability: Option to check acceptable range of the
                    values of parameters after combining two objects.

            Returns:
                ins: if not inplace, a new INS object

        """
        if inplace:
            this = self
        else:
            this = self.copy()

        mask_uvf_this = this._make_mask_copy()
        mask_uvf_other = other._make_mask_copy()

        mask_uvf = super(INS, mask_uvf_this).__add__(mask_uvf_other,
                                                     inplace=False,
                                                     axis=axis,
                                                     run_check=run_check,
                                                     check_extra=check_extra,
                                                     run_check_acceptability=run_check_acceptability)

        if inplace:
            super(INS, this).__add__(other, inplace=True, axis=axis,
                                     run_check=run_check,
                                     check_extra=check_extra,
                                     run_check_acceptability=run_check_acceptability)
        else:
            this = super(INS, this).__add__(other, inplace=False, axis=axis,
                                            run_check=run_check,
                                            check_extra=check_extra,
                                            run_check_acceptability=run_check_acceptability)

        this.metric_array.mask = np.copy(mask_uvf.flag_array)
        this.metric_ms = this.mean_subtract()
        this.sig_array = np.ma.copy(this.metric_ms)

        if not inplace:
            return this

    def _make_mask_copy(self):
        """
        Makes a new INS in flag mode that copies self whose flags are the mask of
        self. Useful for holding the mask temporarily during concatenation etc.

        Returns:
            mask_uvf_copy: A copy of self in flag_mode that holds the mask in
                its flag_array
        """
        mask_uvf_copy = self.copy()
        mask_uvf_copy.to_flag()
        mask_uvf_copy.flag_array = np.copy(self.metric_array.mask)

        return(mask_uvf_copy)
