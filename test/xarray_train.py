#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
import xarray as xr

def plog(text):
    print(text)


poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
poi = poi[poi['competitor']==0]
poi = poi.groupby("id_poi").first().reset_index()
vist = pd.read_csv(baseDir + "raw/tank/ref_vist_h.csv.gz")
dist = vist.pivot_table(index="id_poi",columns="time",values="ref",aggfunc=np.sum)
pist = pd.DataFrame(index=dist.index)
pist.loc[:,"id_poi"] = pist.index
pist = pd.merge(pist,poi,on="id_poi",how="left")

vist = xr.Dataset(data_vars={'ref': (('location', 'time'), dist.values)}
                             ,coords={
                                 'time': [datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S") for x in dist.columns.values]
                                 ,'location': dist.index.values
                             })


vist.ref.plot()
plt.show()

air = vist.ref
air1d = air.isel(location=10)
air1d.plot.line('b-')
plt.show()

fig, axes = plt.subplots(ncols=2)
air1d.plot(ax=axes[0])
air1d.plot.hist(ax=axes[1],bins=20)
plt.tight_layout()
plt.show()

air.isel(location=[10,11,12,13]).plot.line(x='time')
plt.show()

air2d = air.isel(time=50)
air2d.plot()
plt.show()

t = air.isel(time=slice(0, 365 * 4, 250))

g_simple_line = t.isel(time=slice(0,None,4)).plot(x='location', hue='location', col='time', col_wrap=3)
plt.show()




ds = xr.tutorial.load_dataset('rasm')
ds.xc.attrs

data = xr.DataArray(np.random.randn(2, 3), coords={'x': ['a', 'b']}, dims=('x', 'y'))




x =[0,1.0,2.0]
y = [0.0,10.0,20.0]
resX=100
resY=100
grid_x, grid_y = np.mgrid[min(x): max(x):1j * resX, min(y): max(y):1j * resY]





fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9,3))
ds.xc.plot(ax=ax1);
ds.yc.plot(ax=ax2);
plt.show()

ds.Tair[0].plot();
plt.show()

plt.figure(figsize=(7,2));
ax = plt.axes(projection=crs.PlateCarree());
ds.Tair[0].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),x='xc', y='yc', add_colorbar=False);
ax.coastlines();
plt.tight_layout();
plt.show()


mist = pd.read_csv(baseDir + "raw/tank/act_vist_year.csv.tar.gz",compression="gzip")
mist.loc[:,"day"]  = mist['time'].apply(lambda x:x[0:10])
dist = mist.pivot_table(index="id_clust",columns="day",values="ref",aggfunc=np.sum)
dist1 = mist.pivot_table(index="id_clust",columns="day",values="act",aggfunc=np.sum)
dist2 = pd.merge(dist,poi,left_index=True,right_on="id_clust",how="left")
times = dist.columns
locations = np.unique(dist.index)
ds = xr.Dataset(data_vars={'act': (('location', 'time'), dist.values),
                 'ref': (('location', 'time'), dist1.values)}
                ,coords={
                    'time': [datetime.datetime.strptime(x,"%Y-%m-%d") for x in dist.columns.values]
                    ,'location': dist.index.values
                    #                    ,'lon':(['x','y'],dist2['x']),'lat':(['x','y'],dist2['y'])
                })
ds.coords['lat'] = (('x', 'y'), lat)
ds.coords['lon'] = (('x', 'y'), lon)


ds.mean(dim='location').to_dataframe().plot()
plt.show()

df = ds.to_dataframe()
sns.pairplot(df.reset_index(),vars=ds.data_vars)
plt.show()

freeze = (ds['act'] - ds['ref'] <= 0).groupby('time.month').mean('time')
freeze.to_pandas().plot()
plt.show()

monthly_avg = ds.resample(time='1MS').mean()
monthly_avg.sel(location='1-1').to_dataframe().plot(style='s-')
plt.show()

climatology = ds.groupby('time.month').mean('time')
anomalies = ds.groupby('time.month') - climatology
anomalies.mean('location').to_dataframe()[['act', 'ref']].plot()
plt.show()

climatology_mean = ds.groupby('time.month').mean('time')
climatology_std = ds.groupby('time.month').std('time')
stand_anomalies = xr.apply_ufunc(lambda x, m, s: (x - m) / s,ds.groupby('time.month'),climatology_mean, climatology_std)
stand_anomalies.mean('location').to_dataframe()[['act', 'ref']].plot()
plt.show()


some_missing = ds.act.sel(time=ds['time.day'] > 15).reindex_like(ds)
filled = some_missing.groupby('time.month').fillna(climatology.act)
both = xr.Dataset({'some_missing': some_missing, 'filled': filled})
df = both.sel(time='2018').mean('location').reset_coords(drop=True).to_dataframe()
df[['filled', 'some_missing']].plot()
plt.show()




from netCDF4 import num2date

import xarray as xr
dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'all_leap': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '366_day': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]}


def leap_year(year, calendar='standard'):
    """Determine if year is a leap year"""
    leap = False
    if ((calendar in ['standard', 'gregorian',
        'proleptic_gregorian', 'julian']) and
        (year % 4 == 0)):
        leap = True
        if ((calendar == 'proleptic_gregorian') and
            (year % 100 == 0) and
            (year % 400 != 0)):
            leap = False
        elif ((calendar in ['standard', 'gregorian']) and
                 (year % 100 == 0) and (year % 400 != 0) and
                 (year < 1583)):
            leap = False
    return leap

def get_dpm(time, calendar='standard'):
    """
    return a array of days per month corresponding to the months provided in `months`
    """
    month_length = np.zeros(len(time), dtype=np.int)
    cal_days = dpm[calendar]
    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, calendar=calendar):
            month_length[i] += 1
    return month_length


ds = xr.tutorial.load_dataset('rasm')
print(ds)
month_length = xr.DataArray(get_dpm(ds.time.to_index(), calendar='noleap'),coords=[ds.time], name='month_length')
weights = month_length.groupby('time.season') / month_length.astype(float).groupby('time.season').sum()
np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))
ds_weighted = (ds * weights).groupby('time.season').sum(dim='time')
print(ds_weighted)
ds_unweighted = ds.groupby('time.season').mean('time')
ds_diff = ds_weighted - ds_unweighted

notnull = pd.notnull(ds_unweighted['Tair'][0])

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14,12))
for i, season in enumerate(('DJF', 'MAM', 'JJA', 'SON')):
    ds_weighted['Tair'].sel(season=season).where(notnull).plot.pcolormesh(
        ax=axes[i, 0], vmin=-30, vmax=30, cmap='Spectral_r',
        add_colorbar=True, extend='both')

    ds_unweighted['Tair'].sel(season=season).where(notnull).plot.pcolormesh(
        ax=axes[i, 1], vmin=-30, vmax=30, cmap='Spectral_r',
        add_colorbar=True, extend='both')

    ds_diff['Tair'].sel(season=season).where(notnull).plot.pcolormesh(
        ax=axes[i, 2], vmin=-0.1, vmax=.1, cmap='RdBu_r',
        add_colorbar=True, extend='both')

    axes[i, 0].set_ylabel(season)
    axes[i, 1].set_ylabel('')
    axes[i, 2].set_ylabel('')

for ax in axes.flat:
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.axis('tight')
    ax.set_xlabel('')

axes[0, 0].set_title('Weighted by DPM')
axes[0, 1].set_title('Equal Weighting')
axes[0, 2].set_title('Difference')

plt.tight_layout()

fig.suptitle('Seasonal Surface Air Temperature', fontsize=16, y=1.02)
plt.show()






print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
