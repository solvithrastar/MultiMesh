from typing import Union, Tuple
import numpy as np
import cmasher as cmr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from lasif.utils import elliptic_to_geocentric_latitude
from matplotlib import gridspec
from obspy.geodetics import locations2degrees
from salvus.mesh.unstructured_mesh import UnstructuredMesh
from multi_mesh.api import interpolate_to_points
from multi_mesh.utils import lat2colat, greatcircle_points, sph2cart


def plot_depth_slice(
    mesh: Union[str, UnstructuredMesh],
    depth_in_km: float,
    num: int,
    lat_extent: Tuple[float, float] = (-90.0, 90.0),
    lon_extent: Tuple[float, float] = (-180.0, 180.0),
    plot_diff_percentage: bool = False,
    cmap="chroma",
    parameter_to_plot: str = "VSV",
    figsize: Tuple[int, int] = (15, 8),
    projection: ccrs = ccrs.Mollweide(),
    coastlines: bool = True,
    borders: bool = False,
    stock_img: bool = False,
    savefig: bool = False,
    figname: str = "earth.png",
    reverse: bool = False,
    zero_center: bool = True,
    title: str = None,
    limits: Tuple[float, float] = None,
):
    """
    Long and beautiful docstring

    :param mesh: Mesh to use to plot
    :type mesh: Union[str, UnstructuredMesh]
    :param depth_in_km: Depth to slice at
    :type depth_in_km: float
    :param num: Number of points in each dimension
    :type num: int
    :param lat_extent: Extent of domain to query, defaults to (-90.0, 90.0)
    :type lat_extent: Tuple, optional
    :param lon_extent: Extent of domain to query, defaults to (-90.0, 90.0)
    :type lon_extent: Tuple, optional
    :param plot_diff_percentage: If you want to plot the lateral deviations
    :type plot_diff_percentage: bool, optional
    :param cmap: We prefer cmasher colormaps so if the one specified is within
        that library, it will be used from there. Otherwise we will use
        colormaps from matplotlib, defaults to "chroma"
    :type Union(str, matplotlib.colors.ListedColormap): str, optional
    :param parameter_to_plot: Which parameter to plot, defaults to VSV
    :type parameter_to_plot: str, optional
    :param figsize: Size of figure, defaults to (15, 8)
    :type figsize: Tuple(int, int), optional
    :param projection: Projection, defaults to ccrs.Mollweide()
    :type projection: ccrs, optional
    :param coastlines: plot coastlines, defaults to True
    :type coastlines: bool, optional
    :param borders: plot country borders, defaults to False
    :type borders: bool, optional
    :param stock_img: Color oceans and continents, defaults to False
    :type stock_img: bool, optional
    :param savefig: Should figure be saved, defaults to False
    :type savefig: bool, optional
    :param figname: Name of figure, defaults to earth.png
    :type figname: str, optional
    :param reverse: If colormap should be reversed, defaults to False
    :type reverse: bool, optional
    :param zero_center: Make sure that colorbar is zero centered. Mostly
        important for the differential plot, defaults to True
    :type zero_center: bool, optional
    :param title: If you want a custom title to your plot, defaults to None
    :type title: str, optional
    :param limits: If you want to fix your colorbar limits, defaults to None
    :type limits: Tuple[float, float]
    """

    if isinstance(cmap, str):
        cmap = _get_colormap(cmap, reverse)

    if isinstance(mesh, str):
        mesh = UnstructuredMesh.from_h5(mesh)

    points = _create_depthslice(
        depth_in_m=depth_in_km * 1000.0,
        num=num,
        lat_extent=lat_extent,
        lon_extent=lon_extent,
    )

    vals = interpolate_to_points(
        mesh=mesh,
        points=points,
        params_to_interp=[parameter_to_plot],
        make_spherical=False,
        geocentric=True,
    ).reshape(num, num)

    if plot_diff_percentage:
        lat_mean = np.mean(vals)
        vals = (vals - lat_mean) / lat_mean * 100.0
        vmax = np.max(np.abs(vals))
        vmin = -vmax
        if vmax < 0.1:  # This is in here for 1D models which are handled badly
            vals = np.zeros_like(vals)
    else:
        zero_center = False

    Y, X = np.meshgrid(
        np.linspace(lat_extent[0], lat_extent[1], num=num),
        np.linspace(lon_extent[0], lon_extent[1], num=num),
    )
    if not zero_center:
        vmax = None
        vmin = None
    if limits is not None:
        vmax = limits[1]
        vmin = limits[0]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    if stock_img:
        ax.stock_img()
    img = ax.pcolormesh(
        X,
        Y,
        vals,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if coastlines:
        ax.coastlines()
    if borders:
        ax.add_feature(cfeature.BORDERS)
    if title is None:
        if plot_diff_percentage:
            ax.set_title(
                f"{parameter_to_plot} deviations at {depth_in_km} km depth"
            )
        else:
            ax.set_title(f"{parameter_to_plot} at {depth_in_km} km depth")
    else:
        ax.set_title(title, fontsize=20)
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    if savefig:
        plt.savefig(figname)
    else:
        plt.show()


def _create_depthslice(
    depth_in_m: float,
    num: int,
    # lat_extent: Tuple(float, float) = tuple((-90.0, 90.0)),
    # lon_extent: Tuple(float, float) = tuple((-180.0, 180.0)),
    lat_extent=(-90.0, 90.0),
    lon_extent=(-180.0, 180.0),
):
    """
    Create a cloud of points in order to interpolate onto

    :param depth_in_m: Depth of slice
    :type depth_in_m: float
    :param num: Number of samples in each dimension
    :type num: int
    :param lat_extent: Extent of slice, defaults to (-90.0, 90.0)
    :type lat_extent: Tuple, optional
    :param lon_extent: Extent of slice, defaults to (-180.0, 180.0)
    :type lon_extent: Tuple, optional
    """
    lat = np.linspace(lat_extent[0], lat_extent[1], num=num)
    lon = np.linspace(lon_extent[0], lon_extent[1], num=num)
    (
        xx,
        yy,
    ) = np.meshgrid(lat, lon)
    return np.array(
        (xx.ravel(), yy.ravel(), np.ones_like(yy).ravel() * depth_in_m)
    ).T


def _get_colormap(cmap: str, reverse: bool):
    """
    Find the correct colormap object based on the input. We always try to
    find cmasher colormaps

    :param cmap: String describing name of colormap
    :type cmap: str
    :param reverse: Should colormap be reversed?
    :type reverse: bool
    """
    cmash_colormaps = dir(cmr.cm)
    if reverse:
        cmap += "_r"
    if cmap in cmash_colormaps:
        cmap = eval(f"cmr.{cmap}")

    return cmap


def create_projection(
    name: str = "default",
    central_longitude: float = 0.0,
    central_latitude: float = 0.0,
    satellite_height: float = 10000000.0,
    lat_extent=(-90.0, 90.0),
    lon_extent=(-180.0, 180.0),
):
    """
    Function which takes in some information and tries to create an
    appropriate projection.

    For global extent we default to Robinson

    For large continental scale we default to Orthographic

    For an even smaller one we default to Mercator

    Implemented Projections:
        * FlatEarth
        * Mercator
        * Mollweide
        * NearsidePerspective
        * Orthographic
        * PlateCarree
        * Robinson

    :param name: Here you can name a projection, if left empty we will find
        an appropriate projection, defaults to default
    :type name: str, optional
    :param central_longitude: Point of view, does not apply to all projections,
        defaults to 0.0
    :type central_longitude: float, optional
    :param central_latitude: Point of view, does not apply to all projections,
        defaults to 0.0
    :type central_latitude: float, optional
    :param satellite_height: Point of view, does not apply to all projections,
        defaults to 10000000.0
    :type satellite_height: float, optional
    """
    import cartopy.crs as ccrs

    lat_diff = lat_extent[1] - lat_extent[0]
    lon_diff = lon_extent[1] - lon_extent[0]
    if name == "default":
        if lat_diff > 160.0:
            return ccrs.Robinson(central_longitude=central_longitude)
        if lon_diff > 180.0:
            return ccrs.Robinson(central_longitude=central_longitude)

        if lat_diff > 90.0:
            return ccrs.Orthographic(
                central_longitude=central_longitude,
                central_latitude=central_latitude,
            )
        if lon_diff > 90.0:
            return ccrs.Orthographic(
                central_longitude=central_longitude,
                central_latitude=central_latitude,
            )
        return ccrs.Mercator(
            central_longitude=central_longitude,
            min_latitude=lat_extent[0],
            max_latitude=lat_extent[1],
        )

    if name.lower() == "flatearth":
        return ccrs.NorthPolarStereo(central_longitude=central_longitude)
    elif name.lower() == "mercator":
        return ccrs.Mercator(
            central_longitude=central_longitude,
            min_latitude=lat_extent[0],
            max_latitude=lat_extent[1],
        )
    elif name.lower() == "mollweide":
        return ccrs.Mollweide(central_longitude=central_longitude)
    elif name.lower() == "nearsideperspective":
        return ccrs.NearsidePerspective(
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            satellite_height=satellite_height,
        )
    elif name.lower() == "orthographic":
        return ccrs.Orthographic(
            central_latitude=central_latitude,
            central_longitude=central_longitude,
        )
    elif name.lower() == "platecarree":
        return ccrs.PlateCarree(central_longitude=central_longitude)
    elif name.lower() == "robinson":
        return ccrs.Robinson(central_longitude=central_longitude)
    else:
        raise ValueError(
            "Projection not implemented, try implementing it in Cartopy"
        )


def plot_cross_section(
    mesh: Union[str, UnstructuredMesh],
    ref_mesh: Union[str, UnstructuredMesh],
    point_1_lat: float = -20,
    point_1_lng: float = 30,
    point_2_lat: float = 20,
    point_2_lng: float = 60,
    max_depth_in_km: float = 2800,
    nrads: int = 201,
    npoints: int = 301,
    filename: str = "cross_section.pdf",
    cmap="fusion",
    reverse: bool = True,
    clim: Tuple[float, float] = (-5, 5),
    param_to_interp: str = "VSV",
    discontinuities_to_plot: list = [410, 660, 1000],
):
    """
    Plots a cross section through the globe between two specified points.

    :param mesh: salvus mesh object or string
    :param ref_mesh: salvus mesh object or string to reference the velocities
    :param point_1_lat: Point 1 Latitude
    :param point_1_lng: Point 1 Longitude
    :param point_2_lat: Point 2 Latitude
    :param point_2_lng: Point 2 Longitude
    :param max_depth_in_km: Maximum depth of the slice in the km
    :param nrads: Number of points to interpolate in the radial direction
    :param npoints: Number of points to interpolate along the greatcircle
    :param filename: name of the file that gets saved
    :param cmap: Name of the colorbar
    :param reverse: Reverse color bar True/False
    :param clim: Colorbar limits. This is a tuple of (min, max)
    :param param_to_interp: Parameter that you want to plot
    :param discontinuities_to_plot: list of discontinuities to plot, pass an
    empty list to plot nothing.
    """

    # change threshold
    class LowerOrthographic(ccrs.Orthographic):
        @property
        def threshold(self):
            return 1e3

    if isinstance(mesh, str):
        mesh = UnstructuredMesh.from_h5(mesh)
    if isinstance(ref_mesh, str):
        ref_mesh = UnstructuredMesh.from_h5(ref_mesh)

    if isinstance(cmap, str):
        cmap = _get_colormap(cmap, reverse)

    r_earth = 6371000
    rads = np.linspace(r_earth - max_depth_in_km * 1000, r_earth, nrads)

    # Generate interpolation coordinates
    a = greatcircle_points(
        point_1_lat, point_1_lng, point_2_lat, point_2_lng, npts=npoints
    )
    lats, lons = a.T

    # convert to spherical lat
    lats_spherical = np.zeros_like(lats)
    for i in range(len(lats)):
        lats_spherical[i] = elliptic_to_geocentric_latitude(lats[i])

    lats = lat2colat(lats_spherical)
    all_colats, _ = np.meshgrid(lats, rads)
    all_lons, all_rads = np.meshgrid(lons, rads)
    x, y, z = sph2cart(
        np.deg2rad(all_colats.flatten()),
        np.deg2rad(all_lons.flatten()),
        all_rads.ravel(),
    )
    points = np.array((x, y, z)).T

    # Interpolate data
    data = interpolate_to_points(
        mesh,
        points=points,
        make_spherical=True,
        params_to_interp=[param_to_interp],
    )
    ref_data = interpolate_to_points(
        mesh=ref_mesh,
        points=points,
        make_spherical=True,
        params_to_interp=[param_to_interp],
    )
    data = (data - ref_data) / ref_data * 100.0
    data = data.reshape(nrads, npoints)

    # Generate 2D grid, data is plotted on a perfect sphere
    degrees = locations2degrees(
        point_1_lat, point_1_lng, point_2_lat, point_2_lng
    )
    all_degrees = np.linspace(-degrees / 2, degrees / 2, npoints)
    y = np.sin(np.deg2rad(90 - all_degrees))
    x = np.cos(np.deg2rad(90 - all_degrees))
    all_x = np.zeros((npoints, len(rads)))
    all_y = np.zeros((npoints, len(rads)))

    for i in range(len(rads)):
        all_x[:, i] = x * rads[i] / 1000
        all_y[:, i] = y * rads[i] / 1000

    fig = plt.figure(dpi=300)
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 3])

    mid_idx = int(len(lats) / 2)
    start_lat = 90 - lats[0]
    end_lat = 90 - lats[-1]
    start_lng = lons[0]
    end_lng = lons[-1]
    mid_lat = 90 - lats[mid_idx]
    mid_lng = lons[mid_idx]

    # Plot projection
    ax = fig.add_subplot(
        spec[0],
        projection=LowerOrthographic(
            central_latitude=mid_lat, central_longitude=mid_lng
        ),
    )

    ax.set_global()
    ax.stock_img()
    ax.coastlines()
    ax.gridlines()

    # Plot great circle, start, mid and endpoint
    plt.plot(
        [point_1_lng, point_2_lng],
        [point_1_lat, point_2_lat],
        color="red",
        transform=ccrs.Geodetic(),
    )
    # ccrs.Geodetic()
    plt.plot(start_lng, start_lat, "bo", transform=ccrs.Geodetic())
    plt.plot(mid_lng, mid_lat, "go", transform=ccrs.Geodetic())
    plt.plot(end_lng, end_lat, "ro", transform=ccrs.Geodetic())

    # Plot cross section
    ax = fig.add_subplot(spec[1])
    plt.pcolormesh(all_x, all_y, data.T, cmap=cmap, shading="auto")

    # Plot dots on cross section
    left_x = all_x[0, -1]
    left_y = all_y[0, -1]
    mid_x = all_x[mid_idx, -1]
    mid_y = all_y[mid_idx, -1]
    right_x = all_x[-1, -1]
    right_y = all_y[-1, -1]
    plt.plot(left_x, left_y, "bo")
    plt.plot(mid_x, mid_y, "go")
    plt.plot(right_x, right_y, "ro")

    plt.colorbar(ax=ax, shrink=0.4)  # , pad=0.02)
    # cbar.set_label('Perturbation to 1D dv/v [%]')
    # fig.colorbar(pcm, ax=ax,position="bottom", shrink=0.4)#, pad=0.02))
    plt.clim(clim[0], clim[1])

    # Plot discontinuities
    for discontinuity in discontinuities_to_plot:
        plt.plot(
            all_x[:, -1] * (6371 - discontinuity) / 6371,
            all_y[:, -1] * (6371 - discontinuity) / 6371,
            "--",
            color="black",
            linewidth=0.5,
        )

    ax.axis("off")
    ax.axis("equal")
    plt.tight_layout()
    fig.savefig(filename)
