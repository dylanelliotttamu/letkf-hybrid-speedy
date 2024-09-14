#!/bin/bash
#=======================================================================
#    This script prepares for new LETKF cycle-run experiment
# and then runs run_cycle.sh with double precision 
#=======================================================================



bash init.sh

nohup bash run_cycle.sh double &
tail -f nohup.out
