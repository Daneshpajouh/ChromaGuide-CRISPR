#!/bin/bash
# Download production results from Narval cluster via persistent SSH socket
# Destination: local results/ directory

# Set local results directory
LOCAL_DIR="/Users/studio/Desktop/PhD/Proposal/results"
mkdir -p "$LOCAL_DIR"

# Perform download using SCP with the control socket
# Using the user-provided path amird for Narval cluster
scp -v -o 'ControlPath=/tmp/narval_socket' -r narval:/home/amird/chromaguide_experiments/results/* "$LOCAL_DIR/"
