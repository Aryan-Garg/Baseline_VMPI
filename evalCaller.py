#!/usr/bin/env python

import os
from tqdm.auto import tqdm 

def call_eval_with(dataset, depthNet):
    os.system(f'./eval.py --dataset {dataset} -dn {depthNet}')


if __name__ == '__main__':
    # NOTE: Find results in: test_results_new
    datasets = ['Kalantari', 'Hybrid', 'TAMULF', 'Stanford']
    for dataset in tqdm(datasets):
        call_eval_with(dataset, "DeepLens")
        call_eval_with(dataset, "DPT")