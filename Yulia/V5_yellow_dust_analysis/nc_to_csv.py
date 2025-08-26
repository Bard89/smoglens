import xarray as xr
import pandas as pd


nc_file = "/home/julia/smoglens/data/data_sfc.nc"
csv_file = "/home/julia/smoglens/data/yellow_dust_sfc.csv"

ds = xr.open_dataset(nc_file)
df = ds.to_dataframe().reset_index()
df.to_csv(csv_file, index=False)

print("CSV saved:", csv_file)
