#!/bin/bash
#=======================================================================
#    This script prepares for new LETKF cycle-run experiment
# and then runs run_cycle_hyrid_faster.sh 
#=======================================================================



bash init.sh

nohup bash run_cycle_hybrid_faster.sh hybrid &
tail -f nohup.out
