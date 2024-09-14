#!/bin/bash
#=======================================================================
#    This script prepares for new LETKF cycle-run experiment
# and then runs run_cycle.sh with double precision 
#=======================================================================



#bash init.sh
bash init_speedy_forecast_1_9.sh 

nohup bash run_cycle_NO_DA.sh double &
tail -f nohup.out
