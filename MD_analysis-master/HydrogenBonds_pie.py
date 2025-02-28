#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:37:11 2025

@author: nikipaspali
"""

import matplotlib.pyplot as plt
import numpy as np

data_file = "D20_D21_HBONDShbonds-details_sorted.dat"
labels = []
occupancies = []

def custom_autopct(pct):
    total = sum(occupancies)
    occupancy = round(pct * total / 100.0, 2)
    return f'{occupancy:.2f}%' if pct > 0 else ''

with open(data_file, 'r') as file:
    next(file)
    for line in file:
        line = line.strip().split('\t')
        donor = line[0]
        acceptor = line[1]
        occupancy = float(line[2].rstrip('%'))

        if occupancy > 3.0:
            label = f"{donor}-{acceptor}"
            labels.append(label)
            occupancies.append(occupancy)

fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.tab20c(np.linspace(0, 1, len(occupancies)))
patches, texts, autotexts = ax.pie(
    occupancies, labels=labels, startangle=140, colors=colors, autopct=custom_autopct
)

for text in texts:
    text.set_fontsize(16)
    text.set_fontweight('bold')

ax.set_title('Hydrogen Bond Occupancy (%)', pad=20, fontsize=18)
plt.axis('equal')
plt.show()
