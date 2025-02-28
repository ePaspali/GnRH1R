#!/bin/bash

# set the full path to the Rosetta database
DATABASE_PATH="//path/to/Rosetta/main/database/"

# specify the options file: given bellow
OPTIONS_FILE="options_flex.inp"

# list of specific input PDB files
INPUT_FILES=("tem_0060.pdb") #replace the input name for all 15 selected template pdbs and repeat this step

# Iterate over the list of input PDB files
for pdb_file in "${INPUT_FILES[@]}"; do
    # set the input PDB file
    INPUT_PDB="-s $pdb_file"

    # set the output prefix based on the input file name
    OUTPUT_PREFIX="-out:prefix flex_$(basename $pdb_file .pdb)"

    # Run the FlexPepDocking command
    /path/to/Rosetta/main/source/bin/FlexPepDocking.static.linuxgccrelease \
        $INPUT_PDB $OUTPUT_PREFIX @$OPTIONS_FILE > dock_60.log
done

##########################################################################
## options_flex.inp ########

-in:file:spanfile 7BR3_PEP.span -membrane:Membed_init -membrane:Mhbond_depth -score:weights membrane_highres -scorefile score.sc -flexPepDocking:lowres_preoptimize -flexPepDocking:pep_refine
-ex1
-ex2aro
-nstruct 1000
-use_input_sc
##########################################################################


