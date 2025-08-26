"""
Request ID 019897b1-0236-45af-820b-cbcd2fb65566
grid 0.75/0.75
Variable 10m u-component of wind, 10m v-component of wind, 2m temperature, Black carbon aerosol optical depth at 550 nm, Dust aerosol optical depth at 550 nm, Land-sea mask, Particulate matter d < 2.5 µm (PM2.5), Particulate matter d < 10 µm (PM10), Surface pressure
Date 2023-07-14/2023-12-31
Time 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00
Geographical area North: 50°, West: 70°, South: 15°, East: 150°
Data format Zipped netCDF (experimental)
"""
import cdsapi
import os
import zipfile

# Request parameters
dataset = "cams-global-reanalysis-eac4"
request = {
    "date": "2023-07-14/2023-12-31",  # Adjust date range if needed
    "time": [
        "00:00", "03:00", "06:00",
        "09:00", "12:00", "15:00",
        "18:00", "21:00"
    ],
    "data_format": "netcdf_zip",
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "black_carbon_aerosol_optical_depth_550nm",
        "dust_aerosol_optical_depth_550nm",
        "land_sea_mask",
        "particulate_matter_2.5um",
        "particulate_matter_10um",
        "surface_pressure"
    ],
    "area": [50, 70, 15, 150]
}

# Output paths
output_dir = "/home/julia/smoglens/data"
os.makedirs(output_dir, exist_ok=True)
zip_path = os.path.join(output_dir, "cams_yellowdust.zip")

# Download
print("Downloading CAMS data...")
client = cdsapi.Client()
client.retrieve(dataset, request, zip_path)
print(f"Download completed: {zip_path}")

# Unzip
print("Unzipping the file...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(output_dir)
print(f"Files extracted to: {output_dir}")

# List NetCDF files
print("NetCDF files:")
for file in os.listdir(output_dir):
    if file.endswith(".nc"):
        print(f"- {file}")
