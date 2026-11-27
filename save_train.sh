#!/bin/bash

TO="$1"

mkdir -p ~/projects/airi-crl/jaxgcrl/prev_experiments/$TO

cp .awl-job-preset.yml ~/projects/airi-crl/jaxgcrl/prev_experiments/$TO/.awl-job-preset.yml
cp awl_run.sh ~/projects/airi-crl/jaxgcrl/prev_experiments/$TO/awl_run.sh
cp train.py ~/projects/airi-crl/jaxgcrl/prev_experiments/$TO/train.py
