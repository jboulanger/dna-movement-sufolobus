import pandas as pd

from pathlib import Path

# list files from the src folder
root = Path("/cephfs2/jparham/")
dst = Path("/cephfs2/jeromeb/userdata/Baum_group/jparham/Analysis8")
dst.mkdir(exist_ok=True)

filelist =  pd.DataFrame.from_records( [
    {'path':x.relative_to(root),
     'name':x.name,
     'condition':'unknown'}
    for x in root.rglob('Crop*/[!.]*.tif')
    ])

print(f"Number of files {len(filelist)}")
print(filelist)
filelist.to_csv("filelist.csv")