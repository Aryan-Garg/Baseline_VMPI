#!/usr/bin/env python

import os

tamulf = 1000
stanford = 4100

with open("./TAMULF+Stanford/train_files.txt", "w+") as f:
    stanford_files = sorted(os.listdir("/media/data/prasan/datasets/LF_datasets/Stanford/train"))
    # print(stanford_files)
    tamulf_files = sorted(os.listdir("/media/data/prasan/datasets/LF_datasets/TAMULF/train"))

    for e in tamulf_files:
        f.write(f"TAMULF/train/{e}\n")

    for e in stanford_files:
        f.write(f"Stanford/train/{e}\n")
