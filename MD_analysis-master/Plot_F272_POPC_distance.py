#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:57:57 2025

@author: nikipaspali
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

data = np.loadtxt('distance_F272_lipid_2ROS1.txt')
time_data = (data[:, 0] + 900) / 1000
distance_data = data[:, 1]

# plot
plt.figure(figsize=(10, 6))
plt.plot(time_data, distance_data, linestyle='-', color='blueviolet', label='Distance Data')
plt.xlabel('Time (μs)', fontsize=36, fontweight='bold')
plt.ylabel('Distance (Å)', fontsize=36, fontweight='bold')
plt.title('F276-POPC243 Distance', fontsize=36)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))
plt.xlim(0.9, 1.1)
plt.ylim(0.1, 30)
plt.grid(True)
plt.tight_layout()

plt.savefig('distance_F276_POPC.pdf')
plt.show()
