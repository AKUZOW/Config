# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 01:54:47 2022

@author: AK
"""

def export_config():
    """
    Export the default configuration
    """
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)