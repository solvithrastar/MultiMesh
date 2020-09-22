from typing import Union, Tuple
import numpy as np
import cmasher as cmr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


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
    stock_img: bool = False,
    savefig: bool = False,
    figname: str = "earth.png",
    reverse: bool = False,
    zero_center: bool = True,
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
    """
    from multi_mesh.api import interpolate_to_points

    if isinstance(cmap, str):
        cmap = _get_colormap(cmap, reverse)

    if isinstance(mesh, str):
        from salvus.mesh.unstructured_mesh import UnstructuredMesh

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
        make_spherical=True,
        geocentric=True,
    ).reshape(num, num)

    if plot_diff_percentage:
        lat_mean = np.mean(vals)
        vals = (vals - lat_mean) / lat_mean * 100.0
        vmax = np.max(np.abs(vals))
        vmin = -vmax

    Y, X = np.meshgrid(
        np.linspace(lat_extent[0], lat_extent[1], num=num),
        np.linspace(lon_extent[0], lon_extent[1], num=num),
    )
    if not zero_center:
        vmax = None
        vmin = None
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
    if plot_diff_percentage:
        ax.set_title(
            f"{parameter_to_plot} deviations at {depth_in_km} km depth"
        )
    else:
        ax.set_title(f"{parameter_to_plot} at {depth_in_km} km depth")
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
