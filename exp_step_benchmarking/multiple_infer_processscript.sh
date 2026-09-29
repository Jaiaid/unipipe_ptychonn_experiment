#!/bin/bash

#!/bin/bash

# number of process
PROC_COUNT=$1
MEM_SHARE_MODE=$2

# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Provided mem share mode and process count makespan will be calculated from exec in this directory"
   echo
   echo "Syntax: start_exp.sh [-m <s/n>] [-h] [-p <proc count>]"
   echo "options:"
   echo "s     give the "
   echo "h     Print Help."
   echo
}

# parse cmd line arg
while getopts "hm:p:t:" option; do
   case $option in
      h) # display Help
         Help;
         exit;;
      m) 
         MEM_SHARE_MODE=${OPTARG};;
      p)
         PROC_COUNT=${OPTARG};;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

bash cleanup.sh

# start the model sharer process
python3 resnet18_GPU_share.py --shared ${MEM_SHARE_MODE} &
P1_PID=$!

until [ -f checkpoint.pth.tar ]
do
     sleep 2
done

for ((i=1;i<PROC_COUNT;i++)); do
    python3 resnet18_infer_throughput.py --shared ${MEM_SHARE_MODE} &
done

python3 resnet18_infer_throughput.py --shared ${MEM_SHARE_MODE} &
P2_PID=$!

# wait until p2 ends
while ps -p ${P2_PID} > /dev/null
do
     sleep 5
done

kill ${P1_PID}

bash cleanup.sh
