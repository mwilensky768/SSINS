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
from setuptools_scm import get_version
from pathlib import Path
from pyuvdata.branch_scheme import branch_scheme

version_str = get_version(Path(__file__).parent.parent, local_scheme=branch_scheme)
__version__ = version_str
