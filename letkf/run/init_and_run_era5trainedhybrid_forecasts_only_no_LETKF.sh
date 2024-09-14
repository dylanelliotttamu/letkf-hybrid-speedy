#!/bin/bash
#=======================================================================
#    This script prepares for new LETKF cycle-run experiment
# and then runs run_cycle.sh with double precision 
#=======================================================================



#bash init.sh
bash init_era5_trained_hybrid_forecast.sh

nohup bash run_cycle_hybrid_faster_NO_DA.sh hybrid &
tail -f nohup.out
