# Analysis of DNA movement in Sufolobus

In this project, the DNA movement in Sufolobus cells during mitosis acquired in fluorescence microscopy with a membrane marker and a DNA marker is quantified. Events are first identified manually and cropped as small 60x60 pixels vignettes from the acquired data. Cells are then segmented using cellpose with the membrane and DNA markers channels and subsequently tracked using trackpy. Finally, motion is estimate using a Lucas and Kanade motion estimation and several metrics are reported:
- average of the frame difference magnitude of the DNA channel in the cell
- average of the norm of the optical flow estimated with a Lucas and Kanade approach
- average of the norm of the momentum (product of the intensity by the displacement)
- divergence of the norm of the momentum
- number of segmented DNA blob
- total area of the DNA blobs
- asymmetry of the DNA blobs

## Data organization

The data are expected to be organized as follow:

```bash
.
├── results
│   ├── filelist.csv
│   ├── result.csv
│   └── result.h5
└── root
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
git clone https://github.com/jboulanger/dna-movement-sufolobus
```
To create an environment, using using pip + venv:
```bash
cd dna-movement-sufolobus
python -m venv  .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```
or use the conda recipe.

## Usage
The analysis can be run in notebooks or using a command line.

### Using a notebook
- 0_Example.ipynb: run an example on a synthetic dataset
- 1_List_files.ipynb: list files and store the list into filelist.csv.
- 2_Process.ipynb: segment and measure the DNA motion in cells.
- 3_Visualization.ipynb: visualize the results saved in the results files.
    
### Using the command line
- List all files at the `ROOTDIR` folder in the `Crop` subfolders.
```bash
python dnasufo.py list --root $ROOTDIR --dst $DSTDIR
```
- Process first of the listed files
```bash
python dnasufo.py process --root $ROOTDIR --dst $DSTDIR --index 0
```
- Use a slurm script to process all files on a HPC (please adjust the folders in the script):
```bash
sbatch -a 0-239 script/run.sh
```
