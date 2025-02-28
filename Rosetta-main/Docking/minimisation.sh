/path/to/Rosetta/main/source/bin/FlexPepDocking.static.linuxgccrel ease \
-database /path/to/Rosetta/main/database/ \
-s 7BR3_PEP_start.pdb \
-in:file:spanfile 7BR3_PEP.span \ -membrane:Membed_init \ -membrane:Mhbond_depth \ -score:weights membrane_highres \ -ex1 \
-ex2aro \ -flexPepDocking:flexpep_score_only \ -scorefile min.score.sc \
-nstruct 10 \
-out:suffix _min
