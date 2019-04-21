import pandas as pd
import numpy as np

file1 = pd.read_csv('diabetes.csv')
file1.dropna()


file2 = pd.read_csv('hear-attack.csv')
file2.dropna()


file3 = pd.read_csv('Toddler Autism dataset July 2018.csv')
file3.dropna()