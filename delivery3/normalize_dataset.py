import numpy as np
import pandas as pd
from sklearn import preprocessing
from math import e
import matplotlib.pyplot as plt

def normalize_data(input_file, output_file):
    df = pd.read_excel(input_file, engine='openpyxl')
    df_t = df.T
    gene_expressions = df_t.iloc[1:]
    gene_expressions_values = gene_expressions.to_numpy()
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(gene_expressions_values)
    gene_ids = df_t.iloc[0]
    df = pd.DataFrame(x_scaled)
    df_t = df.T
    df_t.to_excel(output_file, index=False, header=False)

normalize_data('../dataset/ThyroidCancerControl.xlsx', "ThyroidCancerNotControlNrmalized.xlsx")
normalize_data('../dataset/ThyroidCancerNotExposed.xlsx', "ThyroidCancerNotExposedNormalized.xlsx")
normalize_data('../dataset/ThyroidCancerExposed.xlsx', "ThyroidCancerExposedNormalized.xlsx")