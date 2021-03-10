
import time
import numpy as np
from typing import Union, List
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
        make_spherical=make_spherical,
    )

    end = time.time()
    runtime = end - start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime:.3f} minutes")
    else:
        print(f"Finished in time: {runtime:.3f} seconds")


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