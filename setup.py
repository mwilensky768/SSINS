from __future__ import absolute_import, division, print_function

from setuptools import setup
import os
import json

from SSINS import version

# Make a GIT_INFO file on install
data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('SSINS', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)


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
    'scripts': ['Scripts/Catalog_Gen.py', 'Scripts/Event_Brightness.py',
                'Scripts/Occ_Scatter.py',
                'Scripts/Brightness_Dict_Integrate.py'],
    'version': version.version,
    'package_data': {'SSINS': data_files},
    'install_requires': ['pyuvdata', 'h5py', 'pyyaml'],
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(**setup_args)
