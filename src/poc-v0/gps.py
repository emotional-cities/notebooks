import harp
import tilemapbase
import pandas as pd

def read(fname, origin=None):
    if origin is None:
        origin = harp.HARP_ORIGIN
    gngga = pd.read_csv(fname)
    gngga.columns = [name.replace('Value.', '') for name in gngga.columns]
    gngga['Timestamp'] = origin + pd.to_timedelta(gngga.Timestamp, 's')
    gngga.set_index('Timestamp', inplace=True)
    return gngga

def plot(gngga,ax=None,**kwargs):
    if ax is None:
        return gngga.plot(x='Longitude', y='Latitude',**kwargs)
    else:
        path = [tilemapbase.project(x,y) for x,y in zip(gngga.Longitude, gngga.Latitude)]
        x, y = zip(*path)
        return ax.plot(x,y,**kwargs)

def plotmap(gngga, ax, bounds=None):
    if bounds is None:
        lonmin, lonmax = gngga.Longitude.min(), gngga.Longitude.max()
        latmin, latmax = gngga.Latitude.min(), gngga.Latitude.max()
    else:
        lonmin, lonmax = bounds[0], bounds[1]
        latmin, latmax = bounds[2], bounds[3]
    tiles = tilemapbase.tiles.build_OSM()
    extent = tilemapbase.Extent.from_lonlat(lonmin, lonmax, latmin, latmax)
    extent = extent.to_aspect(4/3).with_scaling(0.5)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plotter = tilemapbase.Plotter(extent, tiles, width=600)
    return plotter.plot(ax)