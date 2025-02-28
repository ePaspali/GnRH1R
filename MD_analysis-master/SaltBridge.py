#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:09:25 2025

@author: nikipaspali
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

markersize = 48

data_files = [
    'saltbr-GLU68_chainP_segnamePROA-LYS71_chainP_segnamePROA.dat',
    'saltbr-GLU90_chainP_segnamePROA-ARG8_chainP_segnamePROB.dat',
    'saltbr-GLU90_chainP_segnamePROA-LYS121_chainP_segnamePROA.dat',
    'saltbr-ASP98_chainP_segnamePROA-ARG38_chainP_segnamePROA.dat',
    'saltbr-ASP98_chainP_segnamePROA-LYS121_chainP_segnamePROA.dat',
    'saltbr-GLU111_chainP_segnamePROA-LYS115_chainP_segnamePROA.dat',
    'saltbr-ASP138_chainP_segnamePROA-ARG75_chainP_segnamePROA.dat',
    'saltbr-ASP138_chainP_segnamePROA-ARG139_chainP_segnamePROA.dat',
    'saltbr-ASP185_chainP_segnamePROA-LYS191_chainP_segnamePROA.dat',
    'saltbr-ASP293_chainP_segnamePROA-ARG299_chainP_segnamePROA.dat',
]

fig, ax1 = plt.subplots(figsize=(10, 8))
ax2 = ax1.twinx()

all_x_values = []
all_y_values = []
durations = []
color_iterator = iter(sns.color_palette("viridis", len(data_files)))
labels = []

for i, data_file in enumerate(data_files):
    with open(data_file, 'r') as file:
        data = [line.split() for line in file.readlines()]
        frames, distances = zip(*[(int(frame), float(distance)) for frame, distance in data])

    # filter data where distance is <= 3.51
    filtered_data = [(frame * 2 / 100, i * markersize + markersize / 2) for frame, distance in zip(frames, distances) if distance <= 3.51]
    
    duration_frames = len(filtered_data)
    duration_ns = duration_frames * 2 / 100

    all_x_values.extend([x for x, _ in filtered_data])
    all_y_values.extend([y for _, y in filtered_data])
    durations.append(duration_ns)

    match = re.match(r'^saltbr-([A-Z]+)(\d+)_chain([A-Z]+)_segname[A-Z]+-([A-Z]+)(\d+)_chain([A-Z]+)_segname[A-Z]+.dat$', data_file)

    if match:
        amino_acid1 = match.group(1).replace('GLU', 'E').replace('ARG', 'R').replace('ASP', 'D').replace('LYS', 'K')
        number1 = match.group(2)
        chain1 = match.group(3)
        amino_acid2 = match.group(4).replace('GLU', 'E').replace('ARG', 'R').replace('ASP', 'D').replace('LYS', 'K')
        number2 = match.group(5)
        chain2 = match.group(6)
        label = f"{amino_acid1[0]}{number1}-{amino_acid2[0]}{number2}"
        labels.append(label)

        color = next(color_iterator)
        ax1.plot([x for x, _ in filtered_data], [y for _, y in filtered_data], '|', markersize=markersize, color=color, label=label)
    else:
        print(f"Skipping invalid filename: {data_file}")

# set axis labels and title
ax1.set_xlabel('Time (ns)', fontsize=20, fontweight='bold')
ax1.set_ylabel('Salt bridge', color='black', fontsize=20, fontweight='bold')
ax1.set_title('ROS-1', fontsize=20, fontweight='bold')

#sSet y-axis ticks and labels
yticks = np.arange(markersize / 2, len(data_files) * markersize, markersize)
ax1.set_yticks(yticks)
ax1.set_yticklabels(labels, fontsize=18)
ax1.set_ylim(min(all_y_values) - markersize / 2, max(all_y_values) + markersize / 2)

# add horizontal grid lines
for ytick in yticks:
    ax1.axhline(y=ytick + markersize / 2, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# set x-axis range and ticks
x_max = max(all_x_values)
ax1.set_xlim(0, x_max)
xticks = np.arange(0, 1101, 100)
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticks, fontsize=18)

# set right y-axis (duration)
yticks_right = yticks
yticklabels_right = [f'{duration:.2f}' for duration in durations]
ax2.set_yticks(yticks_right)
ax2.set_yticklabels(yticklabels_right, fontsize=18)
ax2.set_ylabel('Duration (ns)', color='black', fontsize=20, fontweight='bold')

# adjust layout and show plot
plt.tight_layout()
plt.show()
