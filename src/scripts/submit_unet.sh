#!/bin/bash
# Convenience script for UNet training
# Usage: ./submit_unet.sh [preliminary] [use_pretrained]
# Default: preliminary=0, use_pretrained=1

PRELIMINARY=${1:-0}
USE_PRETRAINED=${2:-1}

echo "Submitting UNet job (preliminary=$PRELIMINARY, pretrained=$USE_PRETRAINED)"
sbatch ../run_job.sh unet $PRELIMINARY $USE_PRETRAINED 