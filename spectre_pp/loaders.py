import os
import xarray as xr
from xmitgcm import open_mdsdataset

import os

def find_years_and_members(root_dir):
    years = []
    year_to_members = {}

    for entry in sorted(os.listdir(root_dir)):
        year_path = os.path.join(root_dir, entry)
        if os.path.isdir(year_path) and entry.isdigit():
            years.append(entry)
            members = []
            for member_entry in sorted(os.listdir(year_path)):
                member_path = os.path.join(year_path, member_entry)
                if os.path.isdir(member_path):
                    members.append(member_entry)
            year_to_members[entry] = members

    return years, year_to_members

def load_member_dataset(member_path, grid_dir, member_id, geometry='cartesian', delta_t=3600, ref_date='2000-01-01 00:00:00'):
    ds = open_mdsdataset(
        data_dir=member_path,
        grid_dir=grid_dir,
        geometry=geometry,
        read_grid=True,
        prefix=['state', 'tracer', 'diag'],
        default_dtype='float32',
        delta_t=delta_t,
        ref_date=ref_date,
    )

    # Expand along new member dimension
    ds = ds.expand_dims(dim={'member': [member_id]})
    return ds

def load_ensemble(root_dir, grid_dir, years, members):
    datasets = []
    for year in years:
        for member in members:
            member_id = f"{year}_{member}"
            member_path = os.path.join(root_dir, str(year), member)
            ds = load_member_dataset(member_path, grid_dir, member_id)
            datasets.append(ds)
    
    # Concatenate along 'member' dimension
    ds_ensemble = xr.concat(datasets, dim='member')
    return ds_ensemble

# Example usage
root_dir = '/path/to/ROOT'
grid_dir = os.path.join(root_dir, 'grid')
years, year_to_members = find_years_and_members(root_dir)
members = year_to_members[years[0]]  # Assuming all years have the same members

ds_ensemble = load_ensemble(root_dir, grid_dir, years, members)

# Example variable
ssh = ds_ensemble['ETAN']  # shape: (member, time, Y, X)
temp = ds_ensemble['THETA']  # shape: (member, time, Z, Y, X)

print(ssh.dims)
