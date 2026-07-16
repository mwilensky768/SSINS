import numpy as np

import os

import scipy.stats as stats

import eavils_utils
from SSINS import INS
from pyuvdata.parameter import UVParameter

import matplotlib.pyplot as plt

#############################################################################################





def reshape_data(ss):
    return ss.data_array.reshape(ss.Ntimes, ss.Nbls, ss.Nfreqs, ss.Npols)




class EAVILS(INS):
    """
    Defines the EAVILS class, a subclass of the INS class.
    
    The INS class is a subclass of UVFlag, which is a member
    of the pyuvdata software package.
    """

    def __init__(self, indata=None,**kwargs):

        """
        initializes the EAVILS class.


        """
        self._divisor = UVParameter(
            "divisor",
            description=(
                "An array containing the divisor used in EAVILS."
            ),
            form=("Ntimes", "Nfreqs", "Npols"),
            expected_type=float,
            required=False,
        )
        self._spectrum = UVParameter(
            "spectrum",
            description=(
                "An array containing the spectrum output from EAVILS."
            ),
            form=("Ntimes", "Nfreqs", "Npols"),
            expected_type=float,
            required=False,
        )

        self._initial_flags = UVParameter(
            "initial_flags",
            description=(
                "An array containing the initial flags for EAVILS "+
                "(frequencies that should be flagged regardless of "+
                " SSINS, e.g. coarse band lines in the Phase 1 & 2 MWA)."
            ),
            form=("Ntimes", "Nfreqs", "Npols"),
            expected_type=bool,
            required=False,
        )
        super().__init__(indata=indata,
                         **kwargs)

    def add_initial_flags(self,
                      input_initial_flags):
       
        self.initial_flags = input_initial_flags
    
    def build_div_and_spectrum(self,
                      divisor_storage_array,ssins_flags=None):
        '''
        Generates the divisor and spectrum for an EAVILS object.
        
        -divisor_storage_array is generated using the 
            build_divisor_storage_array function on an ss object.
            It is a numpy array with dimensions of 
            Ntimes,Ntimes,Nfreq,Npols. It requires 
            two time axes for recalculating the variance.
            
        -ssins_flags generated on an INS object using mask_to_flags()
            can be optionally included to pre-flag SSINS-detected RFI,
            thus excluding flagged data from the divisor calculation
            and subtracted time mean. This can help when trying to
            use eavils to identify new types of RFI in frequency 
            channels where SSINS has already detected something.
        '''

        if self.initial_flags is None:
            self.initial_flags = np.zeros(self.metric_array.shape,dtype=bool)
            
        if ssins_flags is None:
            ssins_flags = np.zeros(self.metric_array.shape,dtype=bool)
            
        combined_mask = np.logical_or(ssins_flags,self.initial_flags)

        #Builds divisor with any flagged times removed from calculation
        self.divisor = construct_divisor(
            divisor_storage_array,
            mask_array = combined_mask,
            output_sqrt = True,
            output_time_dimension=True
        )
        # Dividing by the square root of number of baselines that contribute to each element of the metric array
        self.divisor=self.divisor/np.sqrt(self.weights_array)

        masked_metric = np.ma.masked_array(data=self.metric_array,mask=combined_mask)

        self.spectrum=(self.metric_array - np.mean(masked_metric,axis=0))/self.divisor
        self.spectrum.mask = self.initial_flags
    def write(self,prefix,clobber=True,data_compression='lzf',output_type='data',sep='_'):
        filename = '%s%sEAVILS%s%s.h5' % (prefix, sep, sep, output_type)
        if output_type == 'data':
            self.metric_array = self.metric_array.data
            super(INS,self).write(filename, clobber=clobber, data_compression=data_compression)



def build_divisor_storage_array(ss,save_prefix=None):
    '''
    Builds an array for recalculation of EAVILS denominator when part of the data is flagged

    Inputs:
    -ss; an undiffed ss object
    -save_prefix; If given a save_prefix will save out a file with the prefix. 
        Prefix can contain a save directory.
    Outputs:
    -divisor_storage_array; <|V(t,b,f,p)|*|V(t',b,f,p)|>_b ; 
        where V=visiblity, t and t'=time, b=baseline, f=frequency, p=polarization.
        In other words, the average over baseline of the product of visibility magnitudes for
        each pair of times.
    -Note: as written this needs the number of baselines to remain the same for all times 
        in an observation. It will take some rewriting to be compatible with other array shapes
        where the number of baselines varies in time.
    '''
    
    divisor_storage_array = (
        np.einsum("ijkl,pjkl->ipkl", np.abs(reshape_data(ss)), np.abs(reshape_data(ss)))
        / ss.Nbls
    )
    if not save_prefix is None:
        np.save(f'{save_prefix}_divisor_storage.npy',divisor_storage_array)
    return divisor_storage_array




def construct_divisor(
    divisor_storage_array,
    mask_array=None,
    output_sqrt=True,
    output_time_dimension=True
):
    """
    Allows the calculation of the EAVILS divisor

    Inputs:
    -divisor_storage_array; records products of visibility magnitudes averaged across baselines.
        >This function, as opposed to a traditional variance calculation, allows for remaking
            the divisor when some times are excluded
        >divisor_storage_array should have shape (Ntimes,Ntimes,Nfreqs,Npols).
    -mask_array; should be a boolean array of shape (Ntimes,Nfreqs,Npols) which will have True
        for masks and False for no mask.
    -output_sqrt; This should always be used to calculate the correct divisor, included
        as an option for testing purposes
    -output_time_dimension; expanded along its time dimension which can be helpful for 
        shape compatibility with other arrays using the output_time_dimension=True option
    Outputs:
    -divisor_output; the divisor array, i.e. sqrt( < Var[ V(t,b,f,p) ]_t >_b )
    """
    Ntimes, Ntimes_check, Nfreqs, Npols = divisor_storage_array.shape
    if Ntimes != Ntimes_check:
        raise Exception(
            f"Input array has shape {divisor_storage_array.shape}, "
            "first two indices should have same length and be the number of included times."
        )
    if mask_array is None:
        mask_array = np.zeros((Ntimes, Nfreqs, Npols), dtype=bool)
        
    
    if mask_array.shape != (Ntimes,Nfreqs,Npols):
        raise Exception(
            f"mask_array has dimensions {mask_array.shape} which differs from expected dimensions, {(Ntimes,Nfreqs,Npols)}"
        )
    
    mask_i = mask_array[:, None, :, :]  # Shape: (N_t, 1, N_f, N_p)
    mask_j = mask_array[None, :, :, :]  # Shape: (1, N_t, N_f, N_p)
    mask_array = np.logical_or(mask_i, mask_j)  # Shape: (N_t, N_t, N_f, N_p)

    
    used_Ntimes = Ntimes - np.trace(mask_array,axis1=0,axis2=1)

    
    divisor_storage_array = np.ma.masked_array(data = divisor_storage_array, mask = mask_array)

    
    # This calculates the average variance using the divisor storage array. 
    # In essence, for a given 
    divisor_output = (
        np.trace(
            divisor_storage_array, axis1=0,axis2=1
        )
        / used_Ntimes
        - np.sum(divisor_storage_array, axis=(0, 1))
        / used_Ntimes**2
    ) * (used_Ntimes / (used_Ntimes - 1))
    
    divisor_output = divisor_output.data

    if output_sqrt:
        divisor_output = np.sqrt(divisor_output)
    if output_time_dimension:
        divisor_output = divisor_output * np.full((Ntimes, Nfreqs, Npols), 1)
    return divisor_output



def plot_maker(eavils,ins,pols,output_path,name_prefix,bl_type_tag):
    '''
    Plotting function for an individual observation
    
    Inputs:
    -eavils; an EAVILS object as defined in this module
    -ins; a SSINS INS object for comparison with EAVILS
    -pols; the labels of each polarization, should be in same order as
     the arrays in the EAVILS/INS objects
    -output_path; the folder to save the output pdf to
    -name_prefix; the prefix for the plot, useful to put some identifying information
        about the source observation 
    -bl_type_tag; the type of baselines (cross or auto) that were used, useful for labelling

    Outputs:
    -saves out a pdf file in output_path
    '''
    
    # For each section of the plot, there are a few parameters we need,
    # which we save as lists to allow some modularity.
    
    # The datasets deals with the actual data we want to plot for each section
    datasets = [
        eavils.metric_array,
        eavils.spectrum*eavils.divisor,
        eavils.spectrum,
        eavils.divisor,
        eavils.spectrum,
        ins.metric_ms
    ]
    
    # Titles just gives titles to each section
    titles = [
        "<|V|;bl>",
        "<|V|;bl> - <<|V|;bl>;t>",
        "spectrum",
        "divisor",
        "spectrum histogram",
        "SSINS"
    ]
    
    # cmaps gives color schemes for relevant plots, otherwise just leaves blank as ''
    cmaps = ["viridis", "coolwarm", "coolwarm", "", "","coolwarm"]
    
    # Vlims deals with defined limits for plot scale when desired
    vlims = [None, None, (-5, 5), None, None, (-5, 5)]
    
    # This is just the plot types. ALlowed options are 'im' for image plot,
    # 'line' for a simple line plot, and 'hist' for histogram plots
    plot_types = ["im", "im", "im", "line", "hist", "im"]

    # Builds figure
    row_count = len(datasets)
    col_count = len(pols)

    fig, axs = plt.subplots(
        nrows=row_count,
        ncols=col_count,
        figsize=(col_count * 4, row_count * 2),
        dpi=300,
    )

    fig.suptitle(name_prefix + ", " + bl_type_tag)
    fig.subplots_adjust(top=0.95)
    fig.tight_layout()

    # Loops through each polarization and row as defined by the datasets list, adding subplots
    for pol_ind in range(len(pols)):
        for ind in range(row_count):
            ax = axs[ind, pol_ind]
            data_index = ind
            values = datasets[data_index][:, :, pol_ind]

            if plot_types[ind] == "im":
                default_aspect = len(values[0]) / len(values)
                desired_aspect = 0.5

                if vlims[ind] is None:
                    active_plot = ax.imshow(
                        values,
                        aspect=default_aspect * desired_aspect,
                        cmap=cmaps[ind],
                    )
                else:

                    active_plot = ax.imshow(
                        values,
                        aspect=default_aspect * desired_aspect,
                        cmap=cmaps[ind],
                        vmin=vlims[ind][0],
                        vmax=vlims[ind][1],
                    )

                fig.colorbar(active_plot, orientation="vertical")

                xticks = np.arange(12, len(eavils.freq_array), 50)
                xticklabels = [
                    "%.0f" % (eavils.freq_array[tick] * 10 ** (-6)) for tick in xticks
                ]

                yticks = np.arange(0, len(values), 5)

                time_names = np.round(
                    (np.unique(eavils.time_array) - np.unique(eavils.time_array)[0])
                    * 86400
                )

                yticklabels = ["%.1f" % (time_names[tick]) for tick in yticks]

                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels, fontsize=6)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels, fontsize=6)
            elif plot_types[ind] == "hist":


                freq_mask = ~np.all(ins.metric_array.mask,axis=(0,2))
                ax.hist(
                    values[:, freq_mask].flatten(),
                    bins=100,
                    histtype="step",
                    density=True,
                )

                ax.set_yscale("log")

                m = np.arange(-4, 4.1, 0.1)
                ax.plot(m, stats.norm.pdf(m))

                max_val = np.round(np.max(values[:, freq_mask].flatten()))
                interval = 2 * max_val / 8
                xticks = np.arange(-max_val, max_val + interval, interval)
                xticklabels = xticks
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels, fontsize=6)
                

            elif plot_types[ind] == "line":
                ax.plot(values[0])
                eavils_utils.forceAspect(ax, aspect=0.5)

                xticks = np.arange(12, len(eavils.freq_array), 50)
                xticklabels = [
                    "%.0f" % (eavils.freq_array[tick] * 10 ** (-6)) for tick in xticks
                ]

                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels, fontsize=6)
                ax.set_xlim(0, len(values[0]))

                box = ax.get_position()

                adjust_x = -0.021
                box.x0 = box.x0 + adjust_x
                box.x1 = box.x1 + adjust_x
                ax.set_position(box)

            ax.set_title(f"{pols[pol_ind]}, {titles[ind]}", fontsize=8)

    save_name = os.path.join(
        output_path, f"{name_prefix}_output_spectra_{bl_type_tag}.pdf"
    )

    print(f"saving {save_name}")
    fig.savefig(save_name)
    plt.close("all")


