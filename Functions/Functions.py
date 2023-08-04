# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:18:38 2022

@author: AK
"""

# =============================================================================
# General Functions to use with config file. TBA
# =============================================================================
# Logs
def log(*args):
    """
    Like print, but also prints datatime. Simple function to keep track without implementing full on logging capabilities
    :param *args: paramaters (variables) to pass to the print function
    :return: True
    """
    print(datetime.datetime.now().strftime('%H:%M:%S.%f')[:-4], *args)
    return True

# Read Files
def read_file(file_name, root=config.data_path, encoding='cp1251', usecols=None, dtype=None, sep=',',
              sheet_name=0, skiprows=None, nrows=None, index_col=None, converters=None, quotechar='\"', decimal='.', df_name='df'):

    if file_name.endswith(".csv") | file_name.endswith(".zip") | file_name.endswith(".tsv"):
        df = pd.read_csv(root + file_name, encoding=encoding, dtype=dtype, sep=sep, decimal=decimal,
                         usecols=usecols, error_bad_lines=False, nrows=nrows, index_col=index_col, quotechar=quotechar
                         )
        # log.info('Loaded ' + file_name)
        return df
    elif file_name.endswith(".txt"):
        df = pd.read_csv(root + file_name, encoding=encoding, dtype=dtype, sep=sep,
                         usecols=usecols, error_bad_lines=False, nrows=nrows)
        # log.info('Loaded ' + file_name)
        return df
    elif file_name.endswith(".xlsx") | file_name.endswith(".xls") | file_name.endswith(".xlsm"):
        df = pd.read_excel(root + file_name, dtype=dtype, sheet_name=sheet_name, skiprows=skiprows, converters=converters)
        if usecols is not None:
            df = df[usecols]
        # log.info('Loaded ' + file_name)
        return df
    elif file_name.endswith('hd5'):
        return pd.read_hdf(root + file_name, key=df_name)
    elif file_name.endswith('.p') | file_name.endswith('.pkl'):
        return pd.read_pickle(root + file_name)
    elif file_name.endswith('.f'):
        return pd.read_feather(root + file_name, columns=usecols)
    else:
        file_name = file_name + '.f'
        return pd.read_feather(root + file_name, columns=usecols)
    
# Save files
def save_file(df, file_name, root=config.data_path,
              enc='cp1251', index=False, sep=',', csv_copy=False):
    if file_name.endswith(".csv"):
        df.to_csv(root + file_name, encoding=enc, index=index, sep=sep)
        # log.info(file_name + ' saved')
        return True
    elif file_name.endswith(".xlsx") | file_name.endswith(".xls"):
        df.to_excel(root + file_name, encoding=enc, index=index)
        # log.info(file_name + ' saved')
        return True
    elif file_name.endswith(".hd5"):
        df.to_hdf(root + file_name, key='df', complevel=6, mode='w')
        return True
    elif file_name.endswith(".pkl"):
        df.to_pickle(root + file_name)
        return True
    elif file_name.endswith(".f"):
        df.reset_index(drop=True).to_feather(root + file_name + '.f')
        return True
    elif file_name.endswith(".txt"):
        with open(root + file_name, "w") as text_file:
            text_file.write(df)
    else:
        # log.info(file_name + ' format is not supported, trying CSV')
        df.reset_index(drop=True).to_feather(root + file_name + '.f')
        if csv_copy:
            df.to_csv(root + file_name + '.csv', encoding=enc, index=index, sep=sep)
        # log.info(file_name + '.csv' + ' saved')
        return True


# =============================================================================
# Descriptive Statistics
# =============================================================================
# Counter tables
def desc_tables_count(inputs, normalize=True):
    if normalize:
        print(inputs.value_counts(normalize=True))
    else:
        print(inputs.value_counts())
        
# Crosstabs
def desc_crosstab(inputs_x,inputs_y,normalize=False):
    print(pd.crosstab(inputs_x,inputs_y,normalize=normalize))


# =============================================================================
# Visualization
# =============================================================================

# Histogram
def hist(inputs,title,bins_a,bins_b,bins_c,figsize_x=12,figsize_y=6):
    plt.figure(figsize=(figsize_x,figsize_y))
    plt.title(title)
    plt.hist(inputs,bins=np.arange(bins_a,bins_b,bins_c))    

for i in ['Revenue_abs', 'Funding_abs', 'Valuation_abs']:
    hist(model_getlatka[f'{i}'],f'{i} Histogram',0,1000,50)

# Bars
def bar(inputs_x,inputs_y,title,figsize_x=12,figsize_y=6):
    plt.figure(figsize=(figsize_x,figsize_y))
    plt.title(title)
    sns.barplot(x=inputs_x,y=inputs_y)

for i in ['Revenue_abs','Valuation_abs','Funding_abs']:
    for j in ['Val_bin','Rev_bin','Fun_bin']:
        bar(model_getlatka[f'{j}'],model_getlatka[f'{i}'],f'{i} - {j}')

# Scatter    
def scatter(inputs_x,inputs_y,title,hue = False,figsize_x=12,figsize_y=6):
    plt.figure(figsize=(figsize_x,figsize_y))
    plt.title(title)
    sns.scatterplot(x = inputs_x,
                    y = inputs_y,
                    hue = hue,
                    s=100)
    
for i in ['Revenue_abs','Funding_abs','P/S','Age (Yrs)','Team Size']:
    scatter(model_getlatka[f'{i}'],model_getlatka['Valuation_abs'],f'{i} - Valuation')
    
# Boxplots

# =============================================================================
# Correlations
# =============================================================================
def corr (inputs, target): 
    for i in ['pearson', 'spearman','kendall']:
        d = {}
        result = pd.DataFrame()
        d[f'corr_{i}'] = pd.DataFrame()
        
        d[f'corr_{i}'] = inputs.corr(method = f'{i}').unstack().sort_values(ascending=False)
        d[f'corr_{i}'] = pd.DataFrame(d[f'corr_{i}']).reset_index()
        d[f'corr_{i}'].columns = ['var_1', 'var_2', f'{i}']
        
        if i == 'pearson':
            corr_p = d[f'corr_{i}']
            continue
        elif i == 'spearman':
            corr_s = d[f'corr_{i}']
            continue
        else:
           corr_k = d[f'corr_{i}'] 
        
        # Merge tables
        result = pd.merge(corr_p, corr_s,  
                              how='left', 
                              left_on = ['var_1','var_2'], 
                              right_on = ['var_1','var_2'])
            
        result = pd.merge(result, corr_k,  
                              how='left', 
                              left_on = ['var_1','var_2'], 
                              right_on = ['var_1','var_2'])
        
        return result.loc[result['var_1'] == target,:]
    
corr(model_getlatka_small, 'Valuation_abs')

# =============================================================================
# Excel
# =============================================================================
import pandas as pd
import numpy as np
from datetime import datetime
import inspect
import subprocess

# Column widths
def get_col_widths(df):
    """
    Constructs a list of column widths for set_column()
    :param df: Data Frame to be processed
    :return: list of widths
    """
    df.columns = [str(col) for col in df.columns]
    col_widths = [max([len(str(s)) for s in df[col].values] + [len(col)]) for col in df.columns]
    return col_widths

# Decimal level
def get_series_decimal_level(s):
    """
    Returns optimal number of decimal numbers for a numeric column based on its median value
    :param s: Series to be processed
    :return: Decimal level
    """
    s_min = s.min()
    if s_min >= 5:
        tag = '0'
    elif s_min >= 0.5:
        tag = '0.0'
    elif s_min >= 0.05:
        tag = '0.00'
    elif s_min >= 0.005:
        tag = '0.000'
    else:
        tag = '0.0000'
    return tag

# Column formats
def get_col_formats(df):
    """
    Constructs a list of column formats for set_column()
    :param df: Data Frame to be processed
    :return: list of formats
    """
    col_types = []
    for col in df.columns:
        s = df[col]
        if s.dtype == np.object:
            col_types.append(False)
            continue
        else:
            col_types.append(get_series_decimal_level(s))
            continue
    return col_types

# Write a worksheet
def write_beautiful_df_to_worksheet(df, writer, workbook, worksheet, get_index):
    """
    Writes df to a given worksheet
    :param df: DataFrame to be written
    :param writer: xlsx writer processing parental .xlsx file
    :param worksheet: worksheet
    :param get_index: True if printing of index is required
    :return: True if written successfully
    """

    header_format = workbook.add_format(
        {'bold': True, 'text_wrap': False, 'font_color': '#000000', 'bottom': 1, 'border_color': '#000000'})

    # Writing Data Frame to specified sheet
    df.to_excel(writer, sheet_name=worksheet.name, index=get_index)

    # Setting header format
    for col_num, value in enumerate(df.columns.values):
        if pd.isnull(value):
            worksheet.write(0, col_num + get_index, 0, header_format)
        else:
            worksheet.write(0, col_num + get_index, value, header_format)

    # Setting index format
    if get_index:
        worksheet.set_column(0, 0, 6)

    # Setting column formatting
    widths = get_col_widths(df)
    formats = get_col_formats(df)
    columns = list(df.columns)

    for i, column in enumerate(columns):
        if formats[i]:
            num_format = workbook.add_format()
            num_format.set_num_format(f'#,##{formats[i]}')
            worksheet.set_column(i + get_index, i + get_index, widths[i], num_format)
        else:
            worksheet.set_column(i + get_index, i + get_index, widths[i])

    return True

# Write a title sheet
def write_title(writer, workbook):
    """
    Writes title sheet containing technical info ebout script launch (e.g. time, file, commit)
    :param writer:
    :return: True if written successfully
    """
    worksheet = workbook.add_worksheet('Title')
    writer.sheets['Title'] = worksheet

    worksheet.hide_gridlines(option=2)

    # Writing general info
    worksheet.write_string(0, 0, f"Oliver Wyman", cell_format=workbook.add_format({'bold': True}))
    worksheet.write_string(2, 0, f"This document was created automatically",
                           cell_format=workbook.add_format({'italic': True, 'font_color': '#FF0000'}))

    # Writing time of launch
    now = datetime.now()
    nowstr = now.strftime("%d-%b-%Y (%H:%M:%S. %f)")
    worksheet.write_string(4, 0, f"Script launch time: {nowstr}")

    # Writing actual git commit at the time of script launch
    last_commit = subprocess.check_output(['git', 'describe', '--always']).strip()
    worksheet.write_string(5, 0, f"Last git commit before launch: {last_commit}")

    # Writing parental file through which the script was launched
    frame = inspect.stack()[1]
    worksheet.write_string(6, 0, f"Script launch file: {frame.filename}")
    return True