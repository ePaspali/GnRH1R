#!/bin/bash

# Set the full path to the Rosetta database
DATABASE_PATH="/opt/software/rosetta/2021.16.61629/main/database/"

# Specify the options file
OPTIONS_FILE="options_flex.inp"

# List of specific input PDB files, change with template name
INPUT_FILES=("tem_0053.pdb")

# Iterate over the list of input PDB files
for pdb_file in "${INPUT_FILES[@]}"
do
    # Set the input PDB file
    INPUT_PDB="-s $pdb_file"

    # Set the output prefix based on the input file name
    OUTPUT_PREFIX="-out:prefix flex_$(basename $pdb_file .pdb)"

    # Run the FlexPepDocking command
    /opt/software/rosetta/2021.16.61629/main/source/bin/FlexPepDocking.static.linuxgccrelease $INPUT_PDB $OUTPUT_PREFIX @$OPTIONS_FILE > dock_53.log

    # Optionally, you can collect the results or perform additional processing here
done

