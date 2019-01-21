import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.utils import resample

def power(sample1, sample2, reps, size, alpha):
    count = 0
    for i in range(0,reps):
        mean1 = 0
        mean2 = 0
        sample1new = np.random.choice(a=sample1)
        sample2new = np.random.choice(a=sample2)
        if np.mean(sample2) > np.mean(sample1):
            count = count + 1
    percentage = count / reps
    return percentage