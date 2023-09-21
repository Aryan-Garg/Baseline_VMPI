#!/usr/bin/env python

import os

tamulf = 1000
stanford = 4100

with open("./TAMULF+Stanford/train_files.txt", "w+") as f:
    for i in range(tamulf):
        f.write("TAMULF/train/" + str(i).zfill(4) + ".npy\n")

    for i in range(stanford):
        f.write("Stanford/train/" + str(i).zfill(4) + ".npy\n")
