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
    ├── img001.tif
    ├── img002.tif
    └── img003.tif
```

## Installation
Clone the repository using :
```bash
git clone
```
Create an environment using:
```bash
conda create -f environment.yml
```


## Usage
- 1_List_files.ipynb: list files and store the list into filelist.csv
- 2_Process.ipynb: measure motion in TIF files and save results in a h5 file
- 2_Visualization.ipynb: visualize the results saved in the h5 file.
    

