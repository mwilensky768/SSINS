"""
init file for SSINS
"""

# Filter annoying Cython warnings that serve no good purpose. see numpy#432
# needs to be done before the imports to work properly
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from .incoherent_noise_spectrum import *
from .plot_lib import *
from .match_filter import *
from .sky_subtract import *
from . import version as _version

__version__ = _version.version
