# Analysis of DNA movement in Sufolobus

Measurement of DNA movement in Sufolobus.

Several motion metrics are reported:
- average of the frame difference magnitude of the DNA channel in the cell
- average of the norm of the optical flow estimated with a Lucas and Kanade approach
- average of the norm of the momentum (product of the intensity by the displacement)
- divergence of the norm of the momentum
- number of segmented dna blob
- total area of the dna blobs
- asymmetry of the dna blobs

## Data organization

```bash
.
├── results
│   ├── filelist.csv
│   ├── result.csv
│   └── result.h5
└── source
    ├── Condition1
    │    ├── Crop
    │    │    ├── img001.tif
    │    │    ├── img002.tif
    │    │    └── img003.tif
    |    └── data.tif 
    └── Condition2
         ├── Crop
         │    ├── img001.tif
         │    ├── img002.tif
         │    └── img003.tif
         └── data.tif 
```

## Installation

Clone the repository using :
```bash
git clone
```
To create an environment, using conda/mamba/micromamba environment:
```bash
conda create -f environment.yml
```
Or with pip + venv:
```bash
python -m venv  .venv
source .venv/bin/activate.sh
pip install -e .
```

## Usage
The analysis can be run in notebooks or using a command line.

### Using a notebook
- 1_List_files.ipynb: list files and store the list into filelist.csv.
- 2_Process.ipynb: measure motion in TIF files and save results in a h5 files.
- 2_Visualization.ipynb: visualize the results saved in the h5 files.
    
### Using the command line
- List all files at the `ROOTDIR` folder in the `Crop` subfolders.
```bash
python dnasufo.py list --root $ROOTDIR --dst $DSTDIR
```
- Process first of the listed files
```bash
python dnasufo.py process --root $ROOTDIR --dst $DSTDIR --index 0
```
- Use a slurm script to process all files on a HPC:
```bash
sbatch -a 0-239 run.sh
```