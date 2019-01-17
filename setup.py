from setuptools import setup
import os


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
                'Scripts/Match_Fraction_Dict_Gen.py',
                'Scripts/Match_Fraction.py', 'Scripts/Read_INS.py'],
    'version': '1.0',
    'package_data': {'SSINS': data_files},
    'install_requires': ['pyuvdata'],
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(**setup_args)
