import time
import numpy as np
from typing import Union, Tuple, List
import pathlib

"""
In here we have many interpolation routines. Currently there is quite a bit of
code repetition since most of this was done to solve a specific application
at the time. Hopefully I'll have time one day to make it more abstract.
"""


def query_model(
    coordinates,
    model,
    nelem_to_search=20,
    parameters="TTI",
    model_path="MODEL/data",
    coordinates_path="MODEL/coordinates",
):
    """
    Provide an array of coordinates, returns an array with model parameters
    for each of these coordinates.

    :param coordinates: Array of coordinates
    :type coordinates: np.array
    :param model: Salvus mesh with model stored on it
    :type model: hdf5 salvus mesh file
    :param nelem_to_search: Number of elements to KDtree query, defaults to 20
    :type nelem_to_search: int, optional
    :param model_path: Where are parameters stored?, defaults to "MODEL/data"
    :type model_path: str, optional
    :param coordinates_path: Where are coordinates stored?, defaults to
        "MODEL/coordinates"
    :type coordinates_path: str, optional
    :return: An array of parameters
    :rtype: np.array
    """
    start = time.time()
    from multi_mesh.components.interpolator import query_model

    values = query_model(
        coordinates=coordinates,
        model=model,
        nelem_to_search=nelem_to_search,
        model_path=model_path,
        coordinates_path=coordinates_path,
    )

    end = time.time()
    runtime = end - start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime} minutes")
    else:
        print(f"Finished in time: {runtime} seconds")
    return values


def exodus_2_gll(
    mesh,
    gll_model,
    gll_order=4,
    dimensions=3,
    nelem_to_search=20,
    parameters="TTI",
    model_path="MODEL/data",
    coordinates_path="MODEL/coordinates",
):
    """
    Interpolate parameters between exodus file and hdf5 gll file.
        Only works in 3 dimensions.
    :param mesh: The exodus file
    :param gll_model: The gll file
    :param gll_order: The order of the gll polynomials
    :param dimensions: How many spatial dimensions in meshes
    :param nelem_to_search: Amount of closest elements to consider
    :param parameters: Parameters to be interolated, possible to pass,
        "ISO", "TTI" or a list of parameters.
    """
    start = time.time()
    from multi_mesh.components.interpolator import exodus_2_gll

    exodus_2_gll(
        mesh,
        gll_model,
        gll_order,
        dimensions,
        nelem_to_search,
        parameters,
        model_path,
        coordinates_path,
    )

    end = time.time()
    runtime = end - start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime} minutes")
    else:
        print(f"Finished in time: {runtime} seconds")


def gll_2_gll(
    from_gll,
    to_gll,
    nelem_to_search=20,
    parameters="TTI",
    from_model_path="MODEL/data",
    to_model_path="MODEL/data",
    from_coordinates_path="MODEL/coordinates",
    to_coordinates_path="MODEL/coordinates",
    gradient=False,
    stored_array=None,
):
    """
    Interpolate parameters between two gll models.
    :param from_gll: path to gll mesh to interpolate from
    :param to_gll: path to gll mesh to interpolate to
    :param nelem_to_search: amount of elements to check
    :param parameters: Parameters to be interpolated, possible to pass, "ISO",
        "TTI" or a list of parameters.
    :return: gll_mesh with new model on it
    :param gradient: If this is a gradient to be added to another gradient,
    only put true if you want to add on top of a currently existing gradient.
    :param stored_array: If either the interpolation matrices are stored
    in a directory or you want to save them into a directory, pass the path
    to the directory as this value. Optional
    """
    start = time.time()
    from multi_mesh.components.interpolator import gll_2_gll

    gll_2_gll(
        from_gll=from_gll,
        to_gll=to_gll,
        nelem_to_search=nelem_to_search,
        parameters=parameters,
        from_model_path=from_model_path,
        to_model_path=to_model_path,
        from_coordinates_path=from_coordinates_path,
        to_coordinates_path=to_coordinates_path,
        gradient=gradient,
        stored_array=stored_array,
    )

    end = time.time()
    runtime = end - start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime} minutes")
    else:
        print(f"Finished in time: {runtime} seconds")


def gll_2_gll_layered(
    from_gll: Union[str, pathlib.Path],
    to_gll: Union[str, pathlib.Path],
    layers: Union[str, List[int]],
    nelem_to_search: int = 20,
    parameters: Union[str, List[str]] = "ISO",
    stored_array: Union[str, pathlib.Path] = None,
    make_spherical: bool = False,
):
    """
    Interpolate parameters between two gll models.
    It loads from_gll to memory, looks at the points of the to_gll and
    assembles a list of unique points which it interpolates onto.
    It then reconstructs the point values based on the initial to_gll points
    and saves it to file.
    Currently not stable if the interpolated parameters are not the same as
    the parameters on the mesh to be interpolated from. Recommend interpolating
    all the parameters from the from_gll mesh.

    :param from_gll: path to gll mesh to interpolate from
    :type from_gll: Union[str, pathlib.Path]
    :param to_gll: path to gll mesh to interpolate to
    :type to_gll: Union[str, pathlib.Path]
    :param nelem_to_search: amount of elements to check, defaults to 20
    :type nelem_to_search: int, optional
    :param parameters: Parameters to be interpolated, possible to pass, "ISO",
        "TTI" or a list of parameters.
    :type parameters: Union[str, List[str]]
    :param stored_array: If you want to store the array for future
        interpolations. If the array exists in that path it will be loaded.
        Store elements under elements.npy and coeffs under coeffs.npy
    :type stored_array: Union[str, pathlib.Path], optional
    :param make_spherical: if mesh is non-spherical, this is recommended,
        defaults to False
    :type make_spherical: bool, optional
    """

    start = time.time()
    from multi_mesh.components.interpolator import gll_2_gll_layered

    gll_2_gll_layered(
        from_gll=from_gll,
        to_gll=to_gll,
        layers=layers,
        parameters=parameters,
        nelem_to_search=nelem_to_search,
        stored_array=stored_array,
        make_spherical=make_spherical,
    )

    end = time.time()
    runtime = end - start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime:.3f} minutes")
    else:
        print(f"Finished in time: {runtime:.3f} seconds")


def gll_2_gll_layered_multi(
    from_gll: Union[str, pathlib.Path],
    to_gll: Union[str, pathlib.Path],
    layers: Union[List[int], str] = "nocore",
    nelem_to_search: int = 20,
    parameters: Union[List[str], str] = "all",
    threads: int = None,
    stored_array: Union[str, pathlib.Path] = None,
    make_spherical: bool = False,
):
    """
    Same function as gll_2_gll_layered except parallel
    Interpolate between two meshes paralellizing over the layers

    :param from_gll: Path to a mesh to interpolate from
    :type from_gll: Union[str, pathlib.Path]
    :param to_gll: Path to a mesh to interpolate onto
    :type to_gll: Union[str, pathlib.Path]
    :param layers: Layers to interpolate. Defaults to "nocore"
    :type layers: Union[List[int], str], optional
    :param nelem_to_search: number of elements to search for, defaults to 20
    :type nelem_to_search: int, optional
    :param parameters: parameters to interpolate, defaults to "all"
    :type parameters: Union[List[str], str], optional
    :param threads: Number of threads, defaults to "all"
    :type threads: int, optional
    :param stored_array: If you want to store the array for future
        interpolations. If the array exists in that path it will be loaded.
        Store elements under elements.npy and coeffs under coeffs.npy
    :type stored_array: Union[str, pathlib.Path], optional
    :param make_spherical: If meshes are not spherical, this is recommended,
        defaults to False
    :type make_spherical: bool, optional
    """

    start = time.time()
    from multi_mesh.components.interpolator import gll_2_gll_layered_multi

    gll_2_gll_layered_multi(
        from_gll=from_gll,
        to_gll=to_gll,
        layers=layers,
        parameters=parameters,
        nelem_to_search=nelem_to_search,
        threads=threads,
        stored_array=stored_array,
        make_spherical=make_spherical,
    )

    end = time.time()
    runtime = end - start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime:.3f} minutes")
    else:
        print(f"Finished in time: {runtime:.3f} seconds")


def gll_2_exodus(
    gll_model,
    exodus_model,
    gll_order=4,
    dimensions=3,
    nelem_to_search=20,
    parameters="TTI",
    model_path="MODEL/data",
    coordinates_path="MODEL/coordinates",
    gradient=False,
):
    """
    Interpolate parameters from gll file to exodus model. Currently I only
    need this for visualization. I could maybe make an xdmf file but that would
    be terribly boring so I'll rather do this for now.
    :param gll_model: path to gll_model
    :param exodus_model: path_to_exodus_model
    """
    start = time.time()
    from multi_mesh.components.interpolator import gll_2_exodus

    gll_2_exodus(
        gll_model,
        exodus_model,
        gll_order,
        dimensions,
        nelem_to_search,
        parameters,
        model_path,
        coordinates_path,
        gradient,
    )

    end = time.time()
    runtime = end - start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime} minutes")
    else:
        print(f"Finished in time: {runtime} seconds")


def interpolate_to_points(
    mesh, points, params_to_interp, make_spherical=False, geocentric=False
):
    """
    Maps values from a mesh to predefined points. The points can by xyz or
    geocentric latlondepth

    :param mesh: Salvus mesh
    :type mesh: Union[str, salvus.mesh.unstructured_mesh.UnstructuredMesh]
    :param points: An array of points, xyz or latlondepth
    :type points: numpy.ndarray
    :param params_to_interp: Which parameters to interpolate
    :type params_to_interp: list[str]
    :param make_spherical: If ellipse, should we make it spherical?,
        defaults to False
    :type make_spherical: bool, optional
    :param geocentric: If coords are latlondepth, defaults to False
    :type geocentric: bool, optional
    """
    if geocentric:
        from multi_mesh.utils import latlondepth_to_xyz

        points = latlondepth_to_xyz(points)
    from multi_mesh.components.interpolator import interpolate_to_points

    return interpolate_to_points(
        mesh=mesh,
        points=points,
        params_to_interp=params_to_interp,
        make_spherical=make_spherical,
    )


def interpolate_to_mesh(
    old_mesh, new_mesh, params_to_interp=["VSV", "VSH", "VPV", "VPH"]
):
    """
    Maps both meshes to a sphere and interpolate values
    from old mesh to new mesh for params to interp.
    Returns the original coordinate system

    Values that are not found are given zero
    """
    from multi_mesh.components.interpolator import (
        interpolate_to_points as _interpolate_to_points,
        map_to_sphere,
    )

    if isinstance(old_mesh, str):
        from salvus.mesh.unstructured_mesh import UnstructuredMesh

        old_mesh = UnstructuredMesh.from_h5(old_mesh)
        if isinstance(new_mesh, str):
            new_mesh_path = new_mesh
            new_mesh = UnstructuredMesh.from_h5(new_mesh)

    # store original point locations
    orig_old_elliptic_mesh_points = np.copy(old_mesh.points)
    orig_new_elliptic_mesh_points = np.copy(new_mesh.points)

    # Map both meshes to a sphere
    map_to_sphere(old_mesh)
    map_to_sphere(new_mesh)
    vals = _interpolate_to_points(old_mesh, new_mesh.points, params_to_interp)

    for i, param in enumerate(params_to_interp):
        new_element_nodal_vals = vals[:, i][new_mesh.connectivity]
        new_mesh.attach_field(param, new_element_nodal_vals)
        # new_mesh.element_nodal_fields[param][:] = new_element_nodal_vals

    # Restore original coordinates
    old_mesh.points = orig_old_elliptic_mesh_points
    new_mesh.points = orig_new_elliptic_mesh_points
    new_mesh.write_h5(new_mesh_path)


# Plotting section:


def plot_depth_slice(
    mesh: Union[str, object],
    depth_in_km: float,
    num: int,
    lat_extent: Tuple[float, float] = (-90.0, 90.0),
    lon_extent: Tuple[float, float] = (-180.0, 180.0),
    plot_diff_percentage: bool = False,
    cmap="chroma",
    parameter_to_plot: str = "VSV",
    figsize: Tuple[int, int] = (15, 8),
    # projection: ccrs = ccrs.Mollweide(),
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
    Plot a depth slice of a Salvus Mesh

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
    from multi_mesh.components.plotter import plot_depth_slice

    plot_depth_slice(
        mesh=mesh,
        depth_in_km=depth_in_km,
        num=num,
        lat_extent=lat_extent,
        lon_extent=lon_extent,
        plot_diff_percentage=plot_diff_percentage,
        cmap=cmap,
        parameter_to_plot=parameter_to_plot,
        figsize=figsize,
        projection="Mollweide",
        coastlines=coastlines,
        borders=borders,
        stock_img=stock_img,
        savefig=savefig,
        figname=figname,
        reverse=reverse,
        zero_center=zero_center,
        title=title,
        limits=limits,
    )


def plot_cross_section(
    mesh: Union[str, object],
    point_1_lat: float = -20,
    point_1_lng: float = 30,
    point_2_lat: float = 20,
    point_2_lng: float = 60,
    max_depth_in_km: float = 2800,
    min_depth_in_km: float = 0.0,
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
    :param point_1_lat: Point 1 Latitude
    :param point_1_lng: Point 1 Longitude
    :param point_2_lat: Point 2 Latitude
    :param point_2_lng: Point 2 Longitude
    :param max_depth_in_km: Maximum depth of the slice in the km
    :param min_depth_in_km: Minimum depth of the slice in the km
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

    from multi_mesh.components.plotter import plot_cross_section

    plot_cross_section(
        mesh=mesh,
        point_1_lat=point_1_lat,
        point_1_lng=point_1_lng,
        point_2_lat=point_2_lat,
        point_2_lng=point_2_lng,
        max_depth_in_km=max_depth_in_km,
        min_depth_in_km=min_depth_in_km,
        nrads=nrads,
        npoints=npoints,
        filename=filename,
        cmap=cmap,
        reverse=reverse,
        clim=clim,
        param_to_interp=param_to_interp,
        discontinuities_to_plot=discontinuities_to_plot,
    )


def find_good_projection(
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
    from multi_mesh.components.plotter import create_projection

    return create_projection(
        name=name,
        central_longitude=central_longitude,
        central_latitude=central_latitude,
        satellite_height=satellite_height,
        lat_extent=lat_extent,
        lon_extent=lon_extent,
    )


def extract_regular_grid(
    mesh: Union[str, pathlib.Path],
    parameters: List[str],
    lat_extent: Tuple[float, float, float],
    lon_extent: Tuple[float, float, float],
    depth_extent: Tuple[float, float, float],
    save_to_netcdf: bool = False,
    netcdf_path: Union[str, pathlib.Path] = None,
):
    """
    A function to extract a regular grid xarray dataset with info about mesh
    if save_to_netcdf is False, the dataset object will be returned.

    :param mesh: path to mesh object or mesh itself
    :type mesh: Union[str, pathlib.Path]
    :param parameters: List of parameters
    :type parameters: List[str]
    :param lat_extent: min_latitude, max_latitude, num_points
    :type lat_extent: Tuple[float, float, float]
    :param lon_extent: min_longitude, max_longitude, num_points
    :type lon_extent: Tuple[float, float, float]
    :param depth_extent: min_depth, max_depth in meters, num_points
    :type depth_extent: Tuple[float, float, float]
    :param save_to_netcdf: If dataset should be saved to a netcdf file,
        defaults to False
    :type save_to_netcdf: bool, optional
    :param netcdf_path: Where to save the file, only needed when save_to_netcdf
    :type netcdf_path: Union[str, pathlib.Path], optional
    """
    from multi_mesh.components.interpolator import extract_regular_grid

    ds = extract_regular_grid(
        mesh=mesh,
        parameters=parameters,
        lat_extent=lat_extent,
        lon_extent=lon_extent,
        depth_extent=depth_extent,
    )

    if save_to_netcdf:
        ds.to_netcdf(path=netcdf_path)
    else:
        return ds


def gll_2_gll_layered_multi_two(
    from_gll: Union[str, pathlib.Path],
    to_gll: Union[str, pathlib.Path],
    layers: Union[List[int], str],
    nelem_to_search: int = 30,
    parameters: Union[List[str], str] = "all",
    stored_array: Union[str, pathlib.Path] = None,
    make_spherical: bool = False,
    tolerance: float = 1.05,
):
    """
    Interpolate between two meshes paralellizing over the layers

    :param from_gll: Path to a mesh to interpolate from
    :type from_gll: Union[str, pathlib.Path]
    :param to_gll: Path to a mesh to interpolate onto
    :type to_gll: Union[str, pathlib.Path]
    :param layers: Layers to interpolate.
    :type layers: Union[List[int], str]
    :param nelem_to_search: number of elements to search for, defaults to 20
    :type nelem_to_search: int, optional
    :param parameters: parameters to interpolate, defaults to "all"
    :type parameters: Union[List[str], str], optional
    :param stored_array: If you want to store the array for future
        interpolations. If the array exists in that path it will be loaded.
        Store elements under elements.npy and coeffs under coeffs.npy
    :type stored_array: Union[str, pathlib.Path], optional
    :param make_spherical: If meshes are not spherical, this is recommended,
        defaults to False
    :type make_spherical: bool, optional
    :param tolerance: Tolerance for how far a point may lay outside of an element.
    Defaults to 1.05 (5%)
    :type tolerance: float
    """
    from multi_mesh.components.interpolator import gll_2_gll_layered_multi_two

    start = time.time()
    gll_2_gll_layered_multi_two(
        from_gll=from_gll,
        to_gll=to_gll,
        layers=layers,
        nelem_to_search=nelem_to_search,
        parameters=parameters,
        stored_array=stored_array,
        make_spherical=make_spherical,
        tolerance=tolerance,
    )
    end = time.time()
    runtime = end - start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime} minutes")
    else:
        print(f"Finished in time: {runtime} seconds")
