#!/bin/bash
#=======================================================================
#    This script prepares for new LETKF cycle-run experiment
# and then runs run_cycle.sh with double precision 
#=======================================================================



#bash init.sh
bash init_hybrid_forecast_1_9_1_9_1_9.sh 
#bash init_hybrid_forecast_1_9_1_9_1st_iter.sh

nohup bash run_cycle_hybrid_faster_NO_DA.sh hybrid &
tail -f nohup.out
