# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:34:34 2021

@author: holge
"""

import pandas as pd
import seaborn as sns


datain = pd.read_csv('https://docs.google.com/spreadsheets/d/' + # Link to sheet
            '1ZyhpCjw7es5SsKBrrheV5NVqhIiD0zofJsauQ4hzXxE' + # on docs
            '/export?' + # 
            'gid=1924442466' + # sheet gid
            '&format=csv', # export as csv
             # Set second row as columnnames in data frame
             header=1
            )

sns.relplot(datain["Workers"][0:4], datain["Speedup"][0:4], kind="line")
sns.relplot(datain["Workers"][0:4], datain["Efficiency"][0:4], kind="line")
sns.relplot(datain["Workers"][0:4], datain["Dask-array version"][0:4], kind="line")

datain = pd.read_csv('https://docs.google.com/spreadsheets/d/' + # Link to sheet
            '1ZyhpCjw7es5SsKBrrheV5NVqhIiD0zofJsauQ4hzXxE' + # on docs
            '/export?' + # 
            'gid=1986874008' + # sheet gid
            '&format=csv', # export as csv
             # Set second row as columnnames in data frame
             header=1
            )

sns.relplot(datain["Workers"][0:4], datain["Speedup"][0:4], kind="line")
sns.relplot(datain["Workers"][0:4], datain["Efficiency"][0:4], kind="line")
sns.relplot(datain["Workers"][0:4], datain["Dask-array version"][0:4], kind="line")

datain = pd.read_csv('https://docs.google.com/spreadsheets/d/' + # Link to sheet
            '1ZyhpCjw7es5SsKBrrheV5NVqhIiD0zofJsauQ4hzXxE' + # on docs
            '/export?' + # 
            'gid=1727913357' + # sheet gid
            '&format=csv', # export as csv
             # Set second row as columnnames in data frame
             header=1
            )

sns.relplot(datain["Workers"][0:4], datain["Speedup"][0:4], kind="line")
sns.relplot(datain["Workers"][0:4], datain["Efficiency"][0:4], kind="line")
sns.relplot(datain["Workers"][0:4], datain["Dask-array version"][0:4], kind="line")

datain = pd.read_csv('https://docs.google.com/spreadsheets/d/' + # Link to sheet
            '1ZyhpCjw7es5SsKBrrheV5NVqhIiD0zofJsauQ4hzXxE' + # on docs
            '/export?' + # 
            'gid=1396448453' + # sheet gid
            '&format=csv', # export as csv
             # Set second row as columnnames in data frame
             header=1
            )

sns.relplot(datain["Workers"][0:4], datain["Speedup"][0:4], kind="line")
sns.relplot(datain["Workers"][0:4], datain["Efficiency"][0:4], kind="line")
sns.relplot(datain["Workers"][0:4], datain["Dask-array version"][0:4], kind="line")
