#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:03:16 2025

@author: nikipaspali
"""

import MDAnalysis as mda

trajectory_files = ["D1.dcd", "D2.dcd", "D3.dcd", "D4.dcd", "D5.dcd", "D6.dcd", "D7.dcd", "D8.dcd", "D9.dcd", "D10.dcd", "D11.dcd", "D12.dcd", "D13.dcd", "D14.dcd", "D15.dcd", "D16.dcd", "D17.dcd", "D18.dcd", "D19.dcd", "D20.dcd", "D21.dcd", "D22.dcd", "D23.dcd", "D24.dcd", "D25.dcd", "D26.dcd", "D27.dcd"]
topology_file = "NTER_FREE_SC.psf"

u = mda.Universe(topology_file)

ion = u.select_atoms("(resid 88 and name SOD)")  # residue = u.select_atoms("(resid 319 and name OD2)") 
residue = u.select_atoms("(resid 280 and name HE1)") 

output_file = open("ion_W280_APO_distances.txt", "w")

for trajectory_file in trajectory_files:
    u.load_new(trajectory_file)

    for ts in u.trajectory:
        distance = mda.lib.distances.distance_array(ion.positions, residue.positions)[0][0]
        output_file.write(f"{distance:.3f}\n")

output_file.close()
