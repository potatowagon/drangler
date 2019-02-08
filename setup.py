NAME = 'drangler'
VERSION = '1.0.0'
SUMMARY = 'cg3002'
AUTHOR = 'Sherry Wong, aka potatowagon'

import os
from distutils.core import setup

old_cwd = os.getcwd()
new_cwd = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(new_cwd)

try:
    setup(
        name=NAME,
        version=VERSION,
        description=SUMMARY,
        author=AUTHOR,
        packages=[NAME],
        package_data={NAME: ['*.so', '*.pyd', '*.dll', '*.dll', '*.properties', '*.ini', '*.info']}
    )
finally:
    os.chdir(old_cwd)
