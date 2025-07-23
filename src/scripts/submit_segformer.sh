#!/bin/bash
# Convenience script for SegFormer training
# Usage: ./submit_segformer.sh [preliminary] [use_pretrained]
# Default: preliminary=0, use_pretrained=1

PRELIMINARY=${1:-0}
USE_PRETRAINED=${2:-1}

echo "Submitting SegFormer job (preliminary=$PRELIMINARY, pretrained=$USE_PRETRAINED)"
sbatch ../run_job.sh segformer $PRELIMINARY $USE_PRETRAINED 