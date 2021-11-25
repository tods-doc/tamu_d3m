# Use date-based versioning, year.month.date, without leading zeros
# For the development branch, use *the day after* the last release with a `.dev0` suffix, for example 2019.2.13.dev0
__version__ = '2021.11.24'

__description__ = 'Common code for D3M project'

__author__ = 'DARPA D3M Program'


from d3m import namespace

namespace.setup()
