# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:10:45 2021

@author: holge
"""
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data1 = {"Execution time [s]" : [219,390,668,1190,1435,117,173,281,
                            515,615,86,139,261,289,650,68,104,
                            194,193,392],
        "Workers": [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
        "Images": [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
        "Speedup": [1.00,1.00,1.00,1.00,1.00,1.87,2.25,2.37,
                    2.31,2.33,2.54,2.80,2.56,4.11,2.21,3.21,
                    3.74,3.45,6.17,3.66]
        }

data2 = {"Execution time [s]" : [268.33, 138.16, 106.22, 95.96,
                                 219, 117, 86, 68],
        "Workers": [1,2,3,4,
                    1,2,3,4],
        "Speedup": [1.00,1.87, 2.54, 3.21,
                    1,1.81,2.73,2.93],
        "Version": ["Individual means", "Individual means", "Individual means", "Individual means",
                    "Combined means", "Combined means", "Combined means", "Combined means"]
        }

df1 = pd.DataFrame(data1)

plt.figure()
fig = sns.relplot(data = df1, x="Workers", y="Execution time [s]", hue = "Images", kind = "line")
fig.set(xticks = range(1,5) )
fig.savefig("workers_vs_execution_time.pdf", bbox_inches='tight')

plt.figure()
fig = sns.relplot(data = df1, x="Images", y="Execution time [s]", hue = "Workers", kind = "line")
fig.set(xticks = range(1,6) )
fig.savefig("images_vs_execution_time.pdf", bbox_inches='tight')

df2 = pd.DataFrame(data2)
plt.figure()
fig = sns.relplot(data = df2, x="Workers", y="Execution time [s]", hue = "Version", kind = "line")
fig.set(xticks = range(1,5) )
fig.savefig("individual_vs_combined_means.pdf", bbox_inches='tight')


