#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:31:51 2025

@author: nikipaspali
"""

import os

def count_files_in_clusters(directory_path='./', num_clusters=84):
    cluster_sizes = {}

    for cluster_number in range(1, num_clusters + 1):
        cluster_pattern = f'c.{cluster_number}.'
        cluster_files = [file for file in os.listdir(directory_path) if cluster_pattern in file]
        file_count = len(cluster_files)
        cluster_sizes[cluster_number] = file_count

    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

    with open('cluster_size.txt', 'w') as output_file:
        for cluster_number, file_count in sorted_clusters:
            output_file.write(f'Cluster {cluster_number}: {file_count} files\n')

    print('Cluster sizes written to "cluster_size.txt".')

current_directory = os.getcwd()  # get the current working directory
count_files_in_clusters(current_directory, num_clusters=84)
