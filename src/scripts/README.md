# HPC Job Submission Scripts

This directory contains convenience scripts for submitting training jobs to SLURM-based HPC clusters.

## Usage

### Individual Model Scripts

Each model has its own submission script:

```bash
# Submit UNet training job
./submit_unet.sh [preliminary] [use_pretrained]

# Submit SegFormer training job  
./submit_segformer.sh [preliminary] [use_pretrained]

# Submit DeepLabV3+ training job
./submit_deeplabv3plus.sh [preliminary] [use_pretrained]

# Submit UNet++ training job
./submit_unetplusplus.sh [preliminary] [use_pretrained]

# Submit MiniUNet training job
./submit_miniunet.sh [preliminary] [use_pretrained]
```

### Parameters

- `preliminary`: 0 for full training, 1 for quick preliminary training (optional, default: 0)
- `use_pretrained`: 0 for training from scratch, 1 for using pretrained encoders (optional, default: 1)

### Examples

```bash
# Full training with pretrained encoder (default)
./submit_unet.sh

# Quick training from scratch
./submit_segformer.sh 1 0

# Full training from scratch
./submit_deeplabv3plus.sh 0 0
```

## Direct SLURM Submission

You can also submit jobs directly using the unified script:

```bash
sbatch ../run_job.sh <model_name> <preliminary> <use_pretrained>
```

## Output

Job outputs are saved to `../cluster_printouts/` with the naming pattern:
- `prints_rsw-segmentation_<job_id>.out` - Standard output
- `prints_rsw-segmentation_<job_id>.err` - Standard error

## Resource Allocation

Default resource allocation per job:
- **Time limit**: 5 hours
- **Memory**: 32GB
- **GPU**: 1 GPU
- **CPUs**: 8 cores

Modify `../run_job.sh` to adjust these settings if needed. 