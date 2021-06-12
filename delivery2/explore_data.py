import numpy as np
import pandas as pd
from math import e
import matplotlib.pyplot as plt

df = pd.read_excel('dataset/ThyroidCancerDataAll.xlsx', engine='openpyxl')
df_t = df.T

index_list = ['Gene ID', 'Min_expression', 'Max_expression', 'Avg', 'Std.Dev.', 'Entropy']
results = pd.DataFrame({'info': index_list})

def get_entropy(column):
    vc = pd.Series(column).value_counts(normalize=True, sort=False)
    base = e
    return -(vc*np.log(vc)/np.log(base)).sum()
    
# df_t.shape[1]
for col in range(df_t.shape[1]):
    temp_list = []
    temp_list.append(df_t.iloc[0][col])  # Gene id
    gene_expressions = df_t.iloc[1:][col]
    temp_list.append(gene_expressions.min())  # Min expression
    temp_list.append(gene_expressions.max())  # Max expression
    temp_list.append(gene_expressions.mean())  # Avg
    temp_list.append(gene_expressions.std())  # Std.
    temp_list.append(get_entropy(gene_expressions))  # Entropy
    temp_df = pd.DataFrame(temp_list)
    results['{}'.format(col)] = temp_df

results = results.T.reset_index(drop=True).T
#results.to_excel("results.xlsx", index=False, header=False)

min_exp = results.T.iloc[1:][1]
max_exp = results.T.iloc[1:][2]
avg_exp = results.T.iloc[1:][3]
std_exp = results.T.iloc[1:][4]
entropy_exp = results.T.iloc[1:][5]

min_fig, min_axes = plt.subplots()
min_exp.plot(ax=min_axes)
min_axes.set_title('Minimum Expression Distrubution')
min_axes.set_xlabel('gene #')
min_axes.set_ylabel('min. exp.')
plt.show()

max_fig, max_axes = plt.subplots()
max_exp.plot(ax=max_axes)
max_axes.set_title('Maximum Expression Distrubution')
max_axes.set_xlabel('gene #')
max_axes.set_ylabel('max. exp.')
plt.show()

avg_fig, avg_axes = plt.subplots()
avg_exp.plot(ax=avg_axes)
avg_axes.set_title('Average Expression Distrubution')
avg_axes.set_xlabel('gene #')
avg_axes.set_ylabel('avg. exp.')
plt.show()

std_fig, std_axes = plt.subplots()
std_exp.plot(ax=std_axes)
std_axes.set_title('Standard Deviation of Expression Distrubution')
std_axes.set_xlabel('gene #')
std_axes.set_ylabel('std. exp.')
plt.show()