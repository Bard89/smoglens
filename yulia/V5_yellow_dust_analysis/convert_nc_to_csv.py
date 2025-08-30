import xarray as xr
import pandas as pd

# Open netCDF file
ds = xr.open_dataset("/home/julia/smoglens/data/data_sfc.nc")  # Use absolute path

# Convert to DataFrame
df = ds.to_dataframe().reset_index()

# Save as CSV
df.to_csv("/home/julia/smoglens/data/data_sfc.csv", index=False)

print("CSV saved:", df.head())
