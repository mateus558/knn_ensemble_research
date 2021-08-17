#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os


out_dir = "knn_optm_res"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

data = pd.read_csv("results_qpoptm_multacc.csv", sep=";", skiprows=[0])
data = data.drop('time', axis=1)
print(data.head())

k_groups = data.groupby(by=["k"]).groups
for key, items in k_groups.items():
    k_res = data.iloc[items] 
    k_res.to_csv(out_dir+f"/{key}_multacc_{out_dir}.csv", sep=";", index=False)
