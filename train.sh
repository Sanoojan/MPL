#!/bin/bash

JOB_ID=76033

while true; do
  # Check the status of the job
  STATUS=$(squeue -j ${JOB_ID} -h -o "%T")
  # If the job is running, print a message and exit the loop
  if [ "$STATUS" = "RUNNING" ]; then
    echo "Job is running"
    sleep 5
    screen -r train2 -X stuff "watch -n 5 nvidia-smi \n"
    break
  fi

  # Wait for a few seconds before checking the status again
  sleep 5
done