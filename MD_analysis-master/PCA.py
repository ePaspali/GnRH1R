#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:54:08 2025

@author: nikipaspali
"""

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from collections import Counter
from sklearn.cluster import DBSCAN

traj_files = ['D10.dcd', 'D11.dcd', 'D12.dcd', 'D13.dcd', 'D14.dcd', 'D15.dcd', 'D16.dcd', 'D17.dcd', 'D18.dcd', 'D19.dcd', 'D20.dcd', 'D21.dcd', 'D22.dcd']
traj = md.load(traj_files, top='ROS_1_SC.pdb')

# specify residue range for the peptide or N-ter
peptide_residues = range(1, 17)  # 1,17 and segname PROA was used for PCA of receptor N-terminus (D10-D22 was t>500ns. segname PROB and resid 1, 10 was used for GnRH PCA, D20-D22 was activation trajectories.
selection_string = f'segname PROA and (residue {peptide_residues[0]} to {peptide_residues[-1]})'
peptide_traj = traj.atom_slice(traj.topology.select(selection_string))

# perform PCA
peptide_coords = peptide_traj.xyz.reshape(peptide_traj.n_frames, -1)
pca = PCA(n_components=2)
peptide_pcs = pca.fit_transform(peptide_coords)

# create a 2D KDE
x = peptide_pcs[:, 0]
y = peptide_pcs[:, 1]
kde = gaussian_kde(np.vstack([x, y]))
x_grid, y_grid = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
z = kde(np.vstack([x_grid.ravel(), y_grid.ravel()]))

#################### comment for gpcr residues //// only uncomment for segname PROB selection (peptide) ####################
# perform DBSCAN clustering
# dbscan = DBSCAN(eps=0.1, min_samples=3)
# cluster_labels = dbscan.fit_predict(peptide_pcs)

# count the number of frames in each cluster
# cluster_counter = Counter(cluster_labels)

# sort clusters based on population
# sorted_clusters = sorted(cluster_counter.items(), key=lambda x: x[1], reverse=True)

# print frame ranges of the three most populated clusters
# for i in range(3):
#     label, count = sorted_clusters[i]
#     frames = np.where(cluster_labels == label)[0]
#     print(f'Cluster {label}: Frame range [{frames.min()}, {frames.max()}], Number of frames: {count}')
#########################################################################################################

# plot
fig, axs = plt.subplots(1, 2, figsize=(12, 10), gridspec_kw={'width_ratios': [6, 2]})  # Adjusted figsize and width_ratios

# density plot
im = axs[0].pcolormesh(x_grid, y_grid, z.reshape(x_grid.shape), shading='auto', cmap='viridis')
axs[0].set_xlabel('PCA 1', fontsize=30, fontweight='bold') 
axs[0].set_ylabel('PCA 2', fontsize=30, fontweight='bold') 
axs[0].set_title('ROS-1 N-ter 1 to 17', fontsize=28, fontweight='bold')
axs[0].tick_params(axis='both', which='both', length=8, labelsize=22)
divider = make_axes_locatable(axs[0])
cax = divider.append_axes("bottom", size="5%", pad=1)
cbar = plt.colorbar(im, cax=cax, orientation='horizontal') 
cbar_ticks = [z.min(), z.max()]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(['Low', 'High'])
cbar.ax.set_xlabel('Population Density', fontsize=24, fontweight='bold')
cbar.ax.tick_params(axis='both', which='both', labelsize=22)

# scree plot (variance explained)
variance_explained = pca.explained_variance_ratio_
bars = axs[1].bar(range(1, len(variance_explained) + 1), variance_explained, align='center', color=['forestgreen', 'blue'])
axs[1].set_xlabel('PCA', fontsize=30, fontweight='bold') 
axs[1].set_ylabel('Variance Explained', fontsize=30, fontweight='bold')
axs[1].set_xticks([1, 2])
axs[1].set_yticks([])
axs[1].tick_params(axis='both', which='both', length=8, labelsize=22)
for i, bar in enumerate(bars):
    height = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width()/2., height, '{:.1%}'.format(variance_explained[i]), ha='center', va='bottom', fontsize=22)

plt.tight_layout()
plt.show()
