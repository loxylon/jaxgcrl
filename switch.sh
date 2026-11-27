#!/bin/bash

FROM_WHERE="$1"

cp $1/.awl-job-preset.yml ~/projects/airi-crl/jaxgcrl/.awl-job-preset.yml
cp $1/awl_run.sh ~/projects/airi-crl/jaxgcrl/awl_run.sh
cp $1/train.py ~/projects/airi-crl/jaxgcrl/train.py
