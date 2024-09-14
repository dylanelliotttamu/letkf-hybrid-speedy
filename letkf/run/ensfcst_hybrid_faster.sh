#!/bin/bash
#=======================================================================
# ensfcst.sh
#   This script runs the SPEEDY model with subdirectory $PROC
#=======================================================================

# Input for this shell
SPEEDY=$1
OUTPUT=$2
YMDH=$3
TYMDH=$4
MEM=$5
PROC=$6

if test 5$6 -eq 5
then
    echo "ERROR in ensfcst.sh"
    exit
fi

echo "./imp.exe $YMDH $MEM > out.lis 2> out.lis.2"
mpirun -np 40 ./imp.exe $YMDH $MEM > out.lis 2> out.lis.2
# 40 changed to 20 to test memory issue 4.28.23
exit 0
