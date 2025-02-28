#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:47:15 2025

@author: nikipaspali
"""

import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = '/path/to/file/distances.csv'
df = pd.read_csv(csv_file_path, skiprows=2)

plt.figure(figsize=(40, 20))

# ROS-1
plt.subplot(2, 2, 1)
line1, = plt.plot(df.iloc[:, 0], df.iloc[:, 3], label='ROS-1 #3', color='cadetblue')  # Plot column 0:1 first
line2, = plt.plot(df.iloc[:, 0], df.iloc[:, 2], label='ROS-1 #2', color='darkblue')  # Plot column 2 next
line3, = plt.plot(df.iloc[:, 0], df.iloc[:, 1], label='ROS-1 #1', color='deepskyblue')  # Plot column 3 last
plt.axhline(y=7.9, color='black', linestyle='--', linewidth=6)  # Line at y=8 is inactive crystal
plt.axhline(y=11, color='green', linestyle='--', linewidth=6)  # Line at y=11 is active AF
plt.xlim(0, 1100)
plt.ylim(6, 15)
plt.xticks(range(0, 1101, 100), fontsize=36)
plt.yticks(range(7, 15), fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=36)
plt.axvline(x=1000, color='black', linestyle='-', linewidth=2)
plt.axvline(x=850, color='black', linestyle='-', linewidth=2)
plt.xlabel('Time (ns)', fontsize=40, fontweight='bold')
plt.ylabel('$\mathbf{R139^{3.50}-T265^{6.33}}(Å)$', fontsize=40, fontweight='bold')
plt.legend(handles=[line3, line2, line1], labels=['ROS-1 #1', 'ROS-1 #2', 'ROS-1 #3'], loc='upper left', fontsize=36)

# ROS-2
plt.subplot(2, 2, 2)
plt.plot(df.iloc[:, 0], df.iloc[:, 4], label='ROS-2 #1', color='g')
plt.plot(df.iloc[:, 0], df.iloc[:, 5], label='ROS-2 #2', color='limegreen')
plt.axhline(y=7.9, color='black', linestyle='--', linewidth=6)  # Line at y=8 is inactive crystal
plt.axhline(y=11, color='green', linestyle='--', linewidth=6)  # Line at y=11 is active AF
plt.xlim(0, 1100)
plt.ylim(6, 15)
plt.xticks(range(0, 1101, 100), fontsize=36)
plt.yticks(range(7, 15), fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=36)
plt.xlabel('Time (ns)', fontsize=40, fontweight='bold')
plt.ylabel('$\mathbf{R139^{3.50}-T265^{6.33}}(Å)$', fontsize=40, fontweight='bold')
plt.legend(loc='upper left', fontsize=36)

# GnRH-Gnrh1r
plt.subplot(2, 2, 3)
plt.plot(df.iloc[:, 0], df.iloc[:, 6], label='Undocked GnRH-GnRH1R #1', color='darkslateblue')
plt.plot(df.iloc[:, 0], df.iloc[:, 7], label='Undocked GnRH-GnRH1R #2', color='darkviolet')
plt.axhline(y=7.9, color='black', linestyle='--', linewidth=6)  # Line at y=8 is inactive crystal
plt.axhline(y=11, color='green', linestyle='--', linewidth=6)  # Line at y=11 is active AF
plt.xlim(0, 1100)
plt.ylim(6, 15)
plt.xticks(range(0, 1101, 100), fontsize=36)
plt.yticks(range(7, 15), fontsize=36)
plt.tick_params(axis='both', which='major', labelsize=36)
plt.axvline(x=0, color='black', linestyle='-', linewidth=2)
plt.xlabel('Time (ns)', fontsize=40, fontweight='bold')
plt.ylabel('$\mathbf{R139^{3.50}-T265^{6.33}}(Å)$', fontsize=40, fontweight='bold')
plt.legend(loc='upper left', fontsize=36)

# free
plt.subplot(2, 2, 4)
plt.plot(df.iloc[:, 0], df.iloc[:, 8], label='Apo-GnRH1R #1', color='k')
plt.plot(df.iloc[:, 0], df.iloc[:, 9], label='Apo-GnRH1R #2', color='grey')
plt.axhline(y=7.9, color='black', linestyle='--', linewidth=6)  # Line at y=8 is inactive crystal
plt.axhline(y=11, color='green', linestyle='--', linewidth=6)  # Line at y=11 is active AF
plt.xlim(0, 1100)
plt.ylim(6, 15)
plt.xticks(range(0, 1101, 100), fontsize=36)
plt.yticks(range(7, 15), fontsize=36)
plt.tick_params(axis='both', which='major', labelsize=36)
plt.xlabel('Time (ns)', fontsize=40, fontweight='bold')
plt.ylabel('$\mathbf{R139^{3.50}-T265^{6.33}}(Å)$', fontsize=40, fontweight='bold')
plt.legend(loc='upper left', fontsize=36)

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
plt.savefig('/path/to/file/tms_multisubplots.pdf')
plt.show()
