# from multi_mesh.helpers import load_lib
# from multi_mesh.io.exodus import Exodus
from multi_mesh import utils
from pykdtree.kdtree import KDTree
import h5py
import sys
import time
import numpy as np
import warnings

import salvus.fem

# Buffer the salvus_fem functions, so accessing becomes much faster
for name, func in salvus.fem._fcts:
    # if name == "__GetInterpolationCoefficients__int_n0_1__int_n1_1__int_n2_0__Matrix_Derive" \
    #            "dA_Eigen::Matrix<double, 2, 1>__Matrix_DerivedB_Eigen::Matrix<double, 4, 1>":
    #     GetInterpolationCoefficients3D = func
    if (
        name
        == "__GetInterpolationCoefficients__int_n0_4__int_n1_4__int_n2_4__Matrix_Derive"
        "dA_Eigen::Matrix<double, 3, 1>__Matrix_DerivedB_Eigen::Matrix<double, 125, 1>"
    ):
        GetInterpolationCoefficients3D_order_4 = func
    if (
        name
        == "__GetInterpolationCoefficients__int_n0_2__int_n1_2__int_n2_2__Matrix_Derive"
        "dA_Eigen::Matrix<double, 3, 1>__Matrix_DerivedB_Eigen::Matrix<double, 27, 1>"
    ):
        GetInterpolationCoefficients3D_order_2 = func
    if name == "__InverseCoordinateTransformWrapper__int_n_4__int_d_3":
        InverseCoordinateTransformWrapper3D_4 = func
    if name == "__InverseCoordinateTransformWrapper__int_n_2__int_d_3":
        InverseCoordinateTransformWrapper3D_2 = func
    if (
        name
        == "__GetInterpolationCoefficients__int_n0_4__int_n1_4__int_n2_0__Matrix_Derive"
        "dA_Eigen::Matrix<double, 2, 1>__Matrix_DerivedB_Eigen::Matrix<double, 25, 1>"
    ):
        GetInterpolationCoefficients2D = func
    if name == "__InverseCoordinateTransformWrapper__int_n_4__int_d_2":
        InverseCoordinateTransformWrapper2D = func
    if name == "__CheckHullWrapper__int_n_4__int_d_3":
        CheckHull = func


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
    :param coordinates_path: Where are coordinates stored?, defaults to "MODEL/coordinates"
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
    Interpolate parameters between exodus file and hdf5 gll file. Only works in 3 dimensions.
    :param mesh: The exodus file
    :param gll_model: The gll file
    :param gll_order: The order of the gll polynomials
    :param dimensions: How many spatial dimensions in meshes
    :param nelem_to_search: Amount of closest elements to consider
    :param parameters: Parameters to be interolated, possible to pass, "ISO", "TTI" or a list of parameters.
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
    :param parameters: Parameters to be interpolated, possible to pass, "ISO", "TTI" or a list of parameters.
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
    from_gll,
    to_gll,
    layers="nocore",
    nelem_to_search=20,
    parameters="all",
    stored_array=None,
):

    start = time.time()
    from multi_mesh.components.interpolator import gll_2_gll_layered

    gll_2_gll_layered(
        from_gll=from_gll,
        to_gll=to_gll,
        layers=layers,
        parameters=parameters,
        nelem_to_search=nelem_to_search,
        stored_array=stored_array,
    )

    end = time.time()
    runtime = end - start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime} minutes")
    else:
        print(f"Finished in time: {runtime} seconds")


# Will keep this function for now, while not really knowing the terminology in Salvus
def gll_2_gll_gradients(simulation, master, first=True):
    """
    Interpolate gradient from simulation mesh to master model. All hdf5 format.
    This can be used to sum gradients too, by making first=False
    :param simulation: path to simulation mesh
    :param master: path to master mesh
    :param first: if false the gradient will be summed on top of existing
    gradient
    """
    with h5py.File(simulation, "r") as sim:
        sim_points = np.array(sim["ELASTIC/coordinates"][:], dtype=np.float64)
        sim_data = sim["ELASTIC/data"][:]
        params = sim["ELASTIC/data"].attrs.get("DIMENSION_LABELS")[2].decode()
        params = params[2:-2].replace(" ", "").replace("grad", "").split("|")

    if "RHO" in params:
        params.remove("RHO")
    if "MassMatrix" in params:
        params.remove("MassMatrix")

    sim_centroids = _find_gll_centroids(sim_points, 2)
    # print(sim_centroids.shape)

    sim_centroid_tree = KDTree(sim_centroids)

    nelem_to_search = 25
    master = h5py.File(master, "r+")

    master_points = np.array(
        master["ELASTIC/coordinates"][:], dtype=np.float64
    )
    master_data = master["ELASTIC/data"]

    gll_points = (4 + 1) ** 2
    values = np.zeros(
        shape=[1, master_points.shape[0], len(params), gll_points]
    )

    nearest_element_indices = np.zeros(
        shape=[master_points.shape[0], gll_points, nelem_to_search],
        dtype=np.int64,
    )
    master_params = (
        master["ELASTIC/data"].attrs.get("DIMENSION_LABELS")[2].decode()
    )
    master_params = (
        master_params[2:-2].replace(" ", "").replace("grad", "").split("|")
    )
    index_map = {}
    for i in range(len(master_params)):
        if master_params[i] in params:
            index_map[i] = params.index(master_params[i])

    for i in range(gll_points):
        _, nearest_element_indices[:, i, :] = sim_centroid_tree.query(
            master_points[:, i, :], k=nelem_to_search
        )

    for s in range(master_points.shape[0]):
        # print(f"Element: {s}")
        for i in range(gll_points):
            # print(f"gll point: {i}")
            point = master_points[s, i, :]
            if point[0] < 1.0e5 or point[0] > 1.3e6:
                for key, value in index_map.items():
                    values[0, :, value, i] += master_data[0, :, key, i]
                # values[0, s, :, i] = master_data[0, s, :, i]
                continue
            if point[1] < 1.0e5 or point[1] > 1.3e6:
                for key, value in index_map.items():
                    values[0, :, value, i] += master_data[0, :, key, i]
                # values[0, s, :, i] = master_data[0, s, :, i]
                continue
            element, ref_coord = _check_if_inside_element(
                sim_points, nearest_element_indices[s, i, :], point
            )
            # print(ref_coord)
            coeffs = get_coefficients(4, 4, 0, ref_coord, 2)
            k = 0
            for param in params:
                # print(f"Parameter: {param}")
                values[0, s, k, i] = np.sum(
                    sim_data[0, element, k, :] * coeffs
                )
                k += 1
    if not first:
        for key, value in index_map.items():
            values[0, :, value, :] += master_data[0, :, key, :]

    del master["ELASTIC/data"]
    master.create_dataset("ELASTIC/data", data=values, dtype="f4")
    master["ELASTIC/data"].dims[0].label = "time"
    master["ELASTIC/data"].dims[1].label = "element"
    for i in params:
        i = "grad" + i
    dimstr = "[ " + " | ".join(params) + " ]"
    master["ELASTIC/data"].dims[2].label = dimstr
    master["ELASTIC/data"].dims[3].label = "point"


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

    store original point locations
    orig_old_elliptic_mesh_points = np.copy(old_mesh.points)
    orig_new_elliptic_mesh_points = np.copy(new_mesh.points)

    Map both meshes to a sphere
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


# I'll keep this function for now, might be needed for smoothiepaper revision
def gradient_2_cartesian_exodus(gradient, cartesian, params, first=False):
    """
    Interpolate gradient from 2D smoothiesem and sum on top of
    2D cartesian mesh. Using gll would be ideal but this function
    is now only with exodus.
    :param gradient: path to cartesian mesh to interpolate from
    :param cartesian: path to smoothiesem mesh to interpolate to
    :param params: list of parameters to interpolate
    :param first: If this is the first gradient, it will overwrite fields
    :return: Gradient interpolated to a cartesian mesh.
    """
    lib = load_lib()

    exodus_a = Exodus(gradient, mode="a")
    print(exodus_a.points)
    print(f"Exodus shape: {exodus_a.points.shape}")
    points = np.array(exodus_a.points, dtype=np.float64)
    exodus_a.points = points

    a_centroids = exodus_a.get_element_centroid()

    # The trilinear interpolator only works in 3D so we fool it to think
    # we are working in 3D
    a_centroids = np.concatenate(
        (a_centroids, np.zeros(shape=(a_centroids.shape[0], 1))), axis=1
    )

    centroid_tree = KDTree(a_centroids)

    nelem_to_search = 20
    exodus_b = Exodus(cartesian, mode="a")

    _, nearest_element_indices = centroid_tree.query(
        exodus_b.points, k=nelem_to_search
    )
    nearest_element_indices = np.array(nearest_element_indices, dtype=np.int64)

    npoints = exodus_b.npoint
    enclosing_element_node_indices = np.zeros((npoints, 4), dtype=np.int64)
    weights = np.zeros((npoints, 4))
    connectivity = exodus_a.connectivity[:, :]
    nfailed = lib.triLinearInterpolator(
        nelem_to_search,
        npoints,
        nearest_element_indices,
        np.ascontiguousarray(connectivity, dtype=np.int64),
        enclosing_element_node_indices,
        np.ascontiguousarray(exodus_a.points),
        weights,
        np.ascontiguousarray(exodus_b.points),
    )

    assert nfailed is 0, f"{nfailed} points could not be interpolated"

    for param in params:
        param_a = exodus_a.get_nodal_field(param)
        values = np.sum(
            param_a[enclosing_element_node_indices] * weights, axis=1
        )
        if not first:
            param_b = exodus_b.get_nodal_field(
                param
            )  # Get pre-existing gradient
            values += param_b  # Add new gradient on top of old one
        exodus_b.attach_field(param, np.zeros_like(values))
        exodus_b.attach_field(param, values)


# Keep this one for now, will be removed later
def gradient_2_cartesian_hdf5(gradient, cartesian, first=False):
    """
    Interpolate gradient on to a cartesian mesh, lets make the cartesian mesh
    exodus now to enable smoothing. That's annoying for the next interpolation
    though but ok
    :param gradient: a gll gradient
    :param cartesian: a cartesian exodus mesh
    :param first: If this is the first one to interpolate so it overwrites
    previous fields on the cartesian mesh.
    :return: Cartesian mesh with summed gradients
    """

    with h5py.File(gradient, "r") as grad:
        grad_points = np.array(
            grad["ELASTIC/coordinates"][:], dtype=np.float64
        )
        grad_data = grad["ELASTIC/data"][:]
        params = grad["ELASTIC/data"].attrs.get("DIMENSION_LABELS")[2].decode()
        params = params[2:-2].replace(" ", "").replace("grad", "").split("|")
        params = params[1:-1]

    print(params)

    grad_centroids = _find_gll_centroids(grad_points, 2)
    centroid_tree = KDTree(grad_centroids)

    nelem_to_search = 25
    cartesian = Exodus(cartesian, mode="a")

    _, nearest_element_indices = centroid_tree.query(
        cartesian.points[:, :2], k=nelem_to_search
    )
    npoints = cartesian.npoint

    values = np.zeros(shape=[npoints, len(params)])
    scaling_factor = 1.0  # 34825988.0

    s = 0
    for point in cartesian.points[:, :2]:
        element, ref_coord = _check_if_inside_element(
            grad_points, nearest_element_indices[s, :], point
        )
        if element is None and ref_coord is None:
            k = 0
            for param in params:
                values[s, k] = 0.0
                k += 1
            s += 1
            continue

        coeffs = get_coefficients(4, 4, 0, ref_coord, 2)
        k = 0
        for param in params:
            values[s, k] = (
                np.sum(grad_data[0, element, k + 1, :] * coeffs)
                * scaling_factor
            )  # I do a k+1 because I'm not using RHO
            k += 1

        s += 1
    i = 0
    for param in params:
        if not first:
            prev_field = cartesian.get_nodal_field(param)
            values[:, i] += prev_field
        print(param)
        cartesian.attach_field(param, np.zeros_like(values[:, i]))
        cartesian.attach_field(param, values[:, i])
        i += 1


# These functions will be removed when I can properly clean this up.
def sum_exodus_fields(collection_mesh, added_mesh, components, first=True):
    from multi_mesh.io.exodus import Exodus

    exodus_a = Exodus(collection_mesh, mode="a")
    exodus_b = Exodus(added_mesh)

    for component in components:
        param = exodus_b.get_nodal_field(component)
        if first:
            exodus_a.attach_field(component, np.zeros_like(param))
        prev_param = exodus_a.get_nodal_field(component)
        param += prev_param
        exodus_a.attach_field(component, param)


def get_coefficients(a, b, c, ref_coord, dimension):
    if dimension == 3:
        if a == 2:
            return GetInterpolationCoefficients3D_order_2(ref_coord)
        else:
            return GetInterpolationCoefficients3D_order_4(ref_coord)
    elif dimension == 2:
        return GetInterpolationCoefficients2D(ref_coord)


def inverse_transform(point, gll_points, dimension):
    if dimension == 3:
        if len(gll_points) == 125:
            return InverseCoordinateTransformWrapper3D_4(
                pnt=point, ctrlNodes=gll_points
            )
        if len(gll_points) == 27:
            return InverseCoordinateTransformWrapper3D_2(
                pnt=point, ctrlNodes=gll_points
            )
    elif dimension == 2:
        return InverseCoordinateTransformWrapper2D(
            pnt=point, ctrlNodes=gll_points
        )
    # return salvus_fem._fcts[29][1](pnt=point, ctrlNodes=gll_points)


def _find_gll_centroids(gll_coordinates, dimensions):
    """
    A function to find the centroid coordinate of gll model
    :param gll: gll model object
    :param dimensions: 1, 2 or 3 dimensions
    :return: array with 3 coordinates per element
    """

    nelements = len(gll_coordinates[:, 0, 0])

    if dimensions != len(gll_coordinates[0, 0, :]):
        raise ValueError("Dimensions of GLL model not the same as input")
    centroids = np.zeros(shape=[nelements, dimensions])

    for d in range(dimensions):
        centroids[:, d] = np.mean(
            gll_coordinates[:, :, d], axis=1, dtype=np.float64
        )

    # print("Found centroids")
    return centroids


def _check_if_inside_element(gll_model, nearest_elements, point, dimension):
    """
    A function to figure out inside which element the point to be interpolated is.
    :param gll: gll model
    :param nearest_elements: nearest elements of the point
    :param point: The actual point
    :return: the Index of the element which point is inside
    """

    point = np.asfortranarray(point, dtype=np.float64)
    ref_coords = np.zeros(len(nearest_elements))
    l = 0
    for element in nearest_elements:
        gll_points = gll_model[element, :, :]
        gll_points = np.asfortranarray(gll_points)
        ref_coord = inverse_transform(
            point=point, gll_points=gll_points, dimension=dimension
        )
        ref_coords[l] = np.sum(np.abs(ref_coord))
        l += 1
        # salvus_fem._fcts[29][1]
        if not np.any(np.abs(ref_coord) > 1.0):
            return element, ref_coord

    warnings.warn(
        "Could not find an element which this points fits into."
        " Maybe you should add some tolerance."
        " Will return the best searched element"
    )
    ind = np.where(ref_coords == np.min(ref_coords))[0][0]
    # ind = ref_coords.index(ref_coords == np.min(ref_coords))
    element = nearest_elements[ind]
    ref_coord = inverse_transform(
        point=point,
        gll_points=np.asfortranarray(
            gll_model[element, :, :], dtype=np.float64
        ),
        dimension=dimension,
    )
    # element = None
    # ref_coord = None

    return element, ref_coord


# to_gll = "/Users/solvi/PhD/workspace/Interpolation/smoothiesem_nlat04.h5"
# from_gll = "/Users/solvi/PhD/workspace/Interpolation/fulastur.h5"
# from_gll = "/Users/solvi/PhD/workspace/Interpolation/Globe3D_csem_60.h5"
# to_gll = "/Users/solvi/PhD/workspace/Interpolation/hressastur.h5"
# gll_2_gll(from_gll, to_gll, nelem_to_search=50, parameters=['RHO', 'VP', 'VS'], from_model_path="MODEL/data", to_model_path="MODEL/data", from_coordinates_path="MODEL/coordinates", to_coordinates_path="MODEL/coordinates", gradient=False)
# mesh = "/Users/solvi/PhD/workspace/Interpolation/Globe3D_csem_50.e"

# exodus_2_gll(mesh, gll_model, gll_order=4, dimensions=3, nelem_to_search=20, parameters="ISO", model_path="MODEL/data", coordinates_path="MODEL/coordinates")
