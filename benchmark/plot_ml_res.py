'''
plot performance of different embedding methods on different tasks
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dat = '/local2/yuyan/gene_emb/data/benchmark/res/summary/all_dis_gene.tsv'
dat = pd.read_csv(dat, sep='\t')

# plot bar plot for each category in go column, value is auprc column, name is emb column
sns.set(style="whitegrid")
sns.set(font_scale=1.5)

# Create the bar plot
g = sns.catplot(x="task", y="auprc", hue="emb", data=dat,
                height=6, kind="bar", palette="muted")

# Customize the plot
g.despine(left=True)
g.set_ylabels("AUPRC")
g.set_xlabels("task")
plt.xticks(rotation=90)
plt.tight_layout()

# Save the plot with the legend to the top-right
plt.savefig('/local2/yuyan/gene_emb/data/benchmark/res/summary/bar_plot_dis_gene.png', dpi=300, bbox_inches='tight')

# Customize and move the legend to the top-right
legend = plt.legend(title="Legend Title", loc="upper right", prop={'size': 6})
legend.get_title().set_fontsize(14) 

# Show the plot
plt.show()
plt.close()


