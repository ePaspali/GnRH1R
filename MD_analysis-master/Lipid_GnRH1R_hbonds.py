#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:01:18 2025

@author: nikipaspali
"""

import os
import matplotlib.pyplot as plt
import numpy as np

output_file_path = '/path/to/directory/sorted_results.txt'
with open(output_file_path, 'r') as output_file:
    lines = output_file.readlines()

x_labels = [line.split('_')[0] for line in lines]
y_values = [float(line.split(': ')[1].strip()) for line in lines]

# filtered indices where y_values > 10 and labels don't start with 'POPC'
filtered_indices = [i for i, value in enumerate(y_values) if value > 10 and not x_labels[i].startswith('POPC')]

# filtered x_labels and y_values
filtered_x_labels = [x_labels[i] for i in filtered_indices]
filtered_y_values = [y_values[i] for i in filtered_indices]

# colour map for the bars
colors = plt.cm.viridis(np.linspace(0, 1, len(filtered_x_labels)))

fig, ax = plt.subplots(figsize=(18, 12))
bars = ax.bar(filtered_x_labels, filtered_y_values, color=colors)

plt.xlabel('Residue', fontsize=32, fontweight='bold')
plt.ylabel('Occupancy (%)', fontsize=28, fontweight='bold')
plt.title('GnRH1R-Membrane Hydrogen bonds', fontsize=32, fontweight='bold')

plt.xticks(rotation=65, ha='center', fontsize=26)
plt.yticks(fontsize=28)

# adding text labels to bars
for bar, value, label in zip(bars, filtered_y_values, filtered_x_labels):
    height = bar.get_height()
    width = bar.get_width()
    x = bar.get_x() + width / 2
    if label in ['R179', 'S55', 'E111']:
        ax.text(x, height + 10, f'{value:.2f}%', ha='center', va='center',
                fontsize=18, bbox=dict(facecolor='white', boxstyle='round,pad=0.3'), rotation=90,
                color='black', edgecolor='black')
    else:
        ax.text(x, height / 2, f'{value:.2f}%', ha='center', va='center',
                fontsize=18, color='black', bbox=dict(facecolor='white', edgecolor='black',
                                                      boxstyle='round,pad=0.3'), rotation=90)

ax.tick_params(axis='both', which='both', direction='out', length=10, width=2)

plt.savefig('/path/to/file/memb_hbonds.pdf', bbox_inches='tight')
plt.show()
