import xarray as xr
from datetime import datetime

your_list = [
    '        3623        3636        3435        1726         363           0',
    '        3637        3620        3426        1719         361           0',
    '        3627        3623        3430        1721         364           0',
    '        3626        3635        3437        1723         358           0',
    '        3603        3616        3435        1730         362           0',
    # ...
    '          74          58          44         170          40           0'
]

# Step 1: Convert list elements into integers
data = [
    [int(value) for value in row.split()]
    for row in your_list
]

# Step 2: Extract and convert datetime values
dates = [
    datetime.strptime(row[0], '%Y%m%d').date()
    for row in data
]

# Step 3: Create xarray dataset
ds = xr.Dataset(
    data_vars={
        'param1': (('time',), [row[1] for row in data]),
        'param2': (('time',), [row[2] for row in data]),
        'param3': (('time',), [row[3] for row in data]),
        'param4': (('time',), [row[4] for row in data]),
    },
    coords={'time': dates}
)

# Print the resulting dataset
print(ds)
