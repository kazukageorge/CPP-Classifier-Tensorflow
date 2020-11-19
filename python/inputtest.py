import numpy as np
import pandas as pd
import datapreparation as dp 

numCells = 2
folderPath = '/home/gokul/Documents/PythonScripts/KangminData/'
dropCols = ['frame']

df_cont, np_cat, np_catNames = dp.prepareData(numCells, folderPath, dropCols)