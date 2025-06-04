import os
import xarray as xr
import xgcm
from xmitgcm import open_mdsdataset
from glob import glob
import warnings
#warnings.filterwarnings('ignore') # Suppress all warnings


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


def discover_ensemble_zarr(root_dir):
    """Returns a dict: {member: {year: [zarr_path, ...]}}"""
    ensemble_map = {}
    for member in sorted(os.listdir(root_dir)):
        member_path = os.path.join(root_dir, member)
        if not os.path.isdir(member_path):
            continue
        year_map = {}
        for year in sorted(os.listdir(member_path)):
            year_path = os.path.join(member_path, year)
            if not os.path.isdir(year_path):
                continue
            zarrs = sorted(glob(os.path.join(year_path, "*.zarr")))
            if zarrs:
                year_map[year] = zarrs
        if year_map:
            ensemble_map[member] = year_map
    return ensemble_map

def load_zarr_ensemble(root_dir):
    ensemble_map = discover_ensemble_zarr(root_dir)
    member_datasets = []

    for member, years in ensemble_map.items():
        year_datasets = []
        for year, zarr_paths in years.items():
            segments = [xr.open_zarr(p, chunks={}) for p in zarr_paths]
            combined = xr.concat(segments, dim='time')
            year_datasets.append(combined)
        member_ds = xr.concat(year_datasets, dim='time')
        member_ds = member_ds.expand_dims({'member': [member]})
        member_datasets.append(member_ds)

    ensemble_ds = xr.concat(member_datasets, dim='member')
    return ensemble_ds



def load_member_dataset(member_path, grid_dir, member_id, geometry='cartesian', delta_t=3600, ref_date='2000-01-01 00:00:00'):
    ds = open_mdsdataset(
        data_dir=member_path,
        grid_dir=grid_dir,
        geometry=geometry,
        read_grid=True,
        prefix=['pickup','pickup_cheapaml'],
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
root_dir = '/tank/spectre/RUNS'
grid_dir = os.path.join(root_dir, 'grid')
#years, year_to_members = find_years_and_members(root_dir)
#members = year_to_members[years[0]]  # Assuming all years have the same members

#ds_ensemble = load_ensemble(root_dir, grid_dir, years, members)
ds_ensemble = load_zarr_ensemble(root_dir)


# Example variable
ssh = ds_ensemble['ETAN']  # shape: (member, time, Y, X)
temp = ds_ensemble['THETA']  # shape: (member, time, Z, Y, X)

print(ssh.dims)
