#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:37:56 2025

@author: nikipaspali
"""

-in:file:l pdb_list
-in:file:fullatom
-cluster:energy_based_clustering:cluster_radius 1.0
-cluster:energy_based_clustering:limit_structures_per_cluster 0
-cluster:energy_based_clustering:cluster_by bb_cartesian
-cluster:energy_based_clustering:use_CB false
-cluster:energy_based_clustering:cyclic false
-cluster:energy_based_clustering:cluster_cyclic_permutations false
-cluster:energy_based_clustering:perform_ABOXYZ_bin_analysis true
-cluster:energy_based_clustering:prerelax
-cluster:energy_based_clustering:relax_rounds 1

#outout clusters distinguasable by names c.1.1.pdb c=cluster .1. = cluster 1 and 1. is structure 1 : naming: <cluster><cluster_number><structure>.pdb

#the commang to run this file is:
/opt/software/rosetta/2021.16.61629/main/source/bin/energy_based_clustering.static.linuxgccrelease @make_cluster.options.inp > cluster.log
# after visualise clusters 


# to make pdb_list: find "$(pwd)" -type f -name 'flex_tem_*.pdb' > pdb_list
