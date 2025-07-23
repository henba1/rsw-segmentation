#!/bin/bash
# Convenience script for MiniUNet training
# Usage: ./submit_miniunet.sh [preliminary] [use_pretrained]
# Default: preliminary=0, use_pretrained=1

PRELIMINARY=${1:-0}
USE_PRETRAINED=${2:-1}

echo "Submitting MiniUNet job (preliminary=$PRELIMINARY, pretrained=$USE_PRETRAINED)"
sbatch ../run_job.sh miniunet $PRELIMINARY $USE_PRETRAINED 