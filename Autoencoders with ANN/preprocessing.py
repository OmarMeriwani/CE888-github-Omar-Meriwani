import pandas as pd
import numpy as np

file1 = pd.read_csv('diabetes.csv')
file1.dropna()


file2 = pd.read_csv('heart-attack.csv')
file2.dropna()


file3 = pd.read_csv('Autism.csv')
file3.dropna()