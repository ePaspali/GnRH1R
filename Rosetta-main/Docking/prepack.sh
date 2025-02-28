#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:40:34 2025

@author: nikipaspali
"""

/path/to/Rosetta/main/source/bin/FlexPepDocking.static.linuxgccrel ease \
-in:file:s 7BR3_PEP_start_min_0003.pdb \
-in:file:spanfile 7BR3_PEP.span \
-membrane:Membed_init \ -membrane::Mhbond_depth \ -score:weights membrane_highres \ -scorefile ppk.sc \ -out:no_nstruct_label \
-out:prefix ppk. \ -flexPepDocking:flexpep_prepack \ -flexPepDocking:flexpep_score_only \ -ex1 \
-ex2aro \
-use_input_sc \ -unboundrot native.pdb \ -nstruct 10
