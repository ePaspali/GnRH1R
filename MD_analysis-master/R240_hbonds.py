#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:54:55 2025

@author: nikipaspali
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

data = np.loadtxt('R240_hbonds.dat')  # Replace 'R240_hbonds.dat' with the actual filename
time = (data[:, 0] / 50.0 + 900.0) / 1000  # Divide the time column by 50 and convert to ns
num_hydrogen_bonds = data[:, 1]

# plot
plt.figure(figsize=(10, 6))
plt.plot(time, num_hydrogen_bonds, linestyle='-', markersize=3, color='dodgerblue')
plt.axhline(y=1.01, color='black', linestyle='-', linewidth=1)  # Black line at y=1
plt.xlabel('Time (μs)', fontsize=36, fontweight='bold') 
plt.ylabel('Hydrogen Bonds', fontsize=36, fontweight='bold') 
plt.title('R240-POPC243 Hydrogen Bonds', fontsize=36) 
plt.grid(True)
plt.xlim(0.9, 1.1)
plt.ylim(0.1, 2)
plt.yticks([1, 2], fontsize=30)
plt.xticks(fontsize=30) 
plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))  # Set x-axis ticks every 0.05
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
plt.tight_layout()
plt.savefig('hbonds_240_POPC.pdf')
plt.show()
