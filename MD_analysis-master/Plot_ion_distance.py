#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:04:11 2025

@author: nikipaspali
"""

import matplotlib.pyplot as plt
import numpy as np

file_path = "/path/to/file/d318_W280_ion__distance.csv"

with open(file_path, 'r') as file:
    lines = file.readlines()

data = []
for line in lines[2:]:
    parts = line.split(',')
    time_ns = float(parts[0]) / 1000  # convert to μs
    distance_d319 = float(parts[1])
    distance_w280 = float(parts[2])
    data.append((time_ns, distance_d319, distance_w280))

x = [entry[0] for entry in data]
y_d319 = [entry[1] for entry in data]
y_w280 = [entry[2] for entry in data]

plt.figure(figsize=(18, 12))
plt.plot(x, y_w280, label='W280-Na$^+$')
plt.plot(x, y_d319, label='D319-Na$^+$')
plt.xlabel('Time (μs)', fontsize=38, fontweight='bold')
plt.ylabel('Distance (Å)', fontsize=38, fontweight='bold')
plt.title('Apo-GnRH1R', fontsize=40, fontweight='bold')
plt.xticks(np.arange(0, 1.2, 0.1), fontsize=36) 
plt.yticks(np.arange(2, 13, 1), fontsize=36)  
plt.legend(fontsize=33)
plt.ylim(1.8, 12)
plt.xlim(0, 1.1)
plt.savefig('ion_D319_w280_distance.pdf')
plt.show()
