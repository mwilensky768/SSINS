from __future__ import absolute_import, division, print_function

from setuptools import setup
import os
import sys
import json


def branch_scheme(version):
    """
    Local version scheme that adds the branch name for absolute reproducibility.

    If and when this is added to setuptools_scm this function and file can be removed.
    """
    if version.exact or version.node is None:
        return version.format_choice("", "+d{time:{time_format}}", time_format="%Y%m%d")
    else:
        if version.branch in ["main", "master"]:
            return version.format_choice("+{node}", "+{node}.dirty")
        else:
            return version.format_choice("+{node}.{branch}", "+{node}.{branch}.dirty")


def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths


data_files = package_files('SSINS', 'data')

setup_args = {
    'name': 'SSINS',
    'author': 'M. Wilensky',
    'url': 'https://github.com/mwilensky768/SSINS',
    'license': 'BSD',
    'description': 'Sky-Subtracted Incoherent Noise Spectra',
    'package_dir': {'SSINS': 'SSINS'},
    'packages': ['SSINS'],
    'include_package_data': True,
    'scripts': ['scripts/Run_HERA_SSINS.py', 'scripts/MWA_EoR_High_Flag.py',
                'scripts/MWA_gpubox_to_SSINS_on_Pawsey.sh', 'scripts/MWA_vis_to_SSINS.py',
                'scripts/occ_csv.py'],
    'package_data': {'SSINS': data_files},
    'setup_requires': ['setuptools_scm'],
    'use_scm_version': {
        "write_to": "SSINS/version.py",
        "local_scheme": branch_scheme,
    },
    'install_requires': ['pyuvdata', 'h5py', 'pyyaml'],
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(**setup_args)
