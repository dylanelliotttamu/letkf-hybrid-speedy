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

# Run
cd $PROC

# Move output
mv ${YMDH}.grd $OUTPUT/anal_f/$MEM
mv ${TYMDH}.grd $OUTPUT/gues/$MEM
#mv out.lis $OUTPUT/gues/$MEM/${TYMDH}out.lis
#mv out.lis.2 $OUTPUT/gues/$MEM/${TYMDH}out.lis.2

exit 0
