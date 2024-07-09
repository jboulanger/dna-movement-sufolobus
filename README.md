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


## Usage
- 1_Process.ipynb: measure motion in TIF files and save results in a h5 file
- 2_Graph.ipynb: visualize the results saved in the h5 file.
    

