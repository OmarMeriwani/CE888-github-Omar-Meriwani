import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

df  =pd.read_csv('vehicles.csv', header=0, sep=',')
#Plot
sns_plot = sns.lmplot(df.columns[0], df.columns[1], data=df, fit_reg=False)

sns_plot.axes[0, 0].set_ylim(0, )
sns_plot.axes[0, 0].set_xlim(0, )

sns_plot.savefig("scaterplot.png", bbox_inches='tight')
sns_plot.savefig("scaterplot.pdf", bbox_inches='tight')
#Histogram
plt.clf()
data = df.values.T[1]
sns_plot2 = sns.distplot(data, bins=20, kde=False, rug=True).get_figure()

axes = plt.gca()
axes.set_xlabel('Fleet value')
axes.set_ylabel('Frequency')

sns_plot2.savefig("histogram1.png", bbox_inches='tight')
sns_plot2.savefig("histogram1.pdf", bbox_inches='tight')
