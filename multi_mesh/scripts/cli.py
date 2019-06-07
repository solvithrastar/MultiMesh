import click
import warnings

from multi_mesh.io.exodus import Exodus

from scipy.spatial import cKDTree
import numpy as np
from multi_mesh.helpers import load_lib
import h5py
import time
from pykdtree.kdtree import KDTree
from multi_mesh import utils

import salvus_fem
# Buffer the salvus_fem functions, so accessing becomes much faster
for name, func in salvus_fem._fcts:
    if name == "__GetInterpolationCoefficients__int_n0_4__int_n1_4__int_n2_4__Matrix_Derive" \
               "dA_Eigen::Matrix<double, 3, 1>__Matrix_DerivedB_Eigen::Matrix<double, 125, 1>":
        GetInterpolationCoefficients = func
    if name == "__InverseCoordinateTransformWrapper__int_n_4__int_d_3":
        InverseCoordinateTransformWrapper = func

warnings.simplefilter(action='ignore', category=FutureWarning)

@click.group()
def cli():
    pass

@cli.command()
@click.option('--mesh_a', help="Salvus continuous exodus file.", required=True)
@click.option('--mesh_b', help="Salvus continuous exodus file.", required=True)
@click.option('--params', help="parameter to interpolate.", required=False,
              type=list)
def interpolate_mesh_a_to_b(mesh_a, mesh_b, params=["TTI"]):
    """
    Interpolates values from mesh A onto mesh B, exodus to exodus.
    Only works for 3D meshes
    Inputs:
    mesh_a: exodus filename with a mesh,
    mesh_b: expdus fileneme with a mesh,
    params: List of parameters to interpolate, if none specified, all TTI
            parameters will be used.
    """
    from multi_mesh.io.exodus import Exodus
    from scipy.spatial import cKDTree
    import numpy as np
    from multi_mesh.helpers import load_lib
    lib = load_lib()

    if params[0] == "TTI":
        params = ["VSH", "VSV", "VPV", "VPH", "RHO", "ETA", "QKAPPA", "QMU"]

    # Read Mesh A (exodus format)
    exodus_a = Exodus(mesh_a)

    # Create KDTree from mesh a element centroids
    a_centroids = exodus_a.get_element_centroid()
    centroid_tree = cKDTree(a_centroids, balanced_tree=False)

    # Read Mesh B and search for nearest_element_indices
    nelem_to_search = 20
    exodus_b = Exodus(mesh_b, mode="a")
    _, nearest_element_indices = centroid_tree.query(exodus_b.points,
                                                     k=nelem_to_search)

    # number of points that require interpolation.
    npoints = exodus_b.npoint
    enclosing_elem_node_indices = np.zeros((npoints, 8), dtype=np.int64)
    weights = np.zeros((npoints, 8))  # initiate interpolation weights
    permutation = [0, 3, 2, 1, 4, 5, 6, 7] # indices of nodes.
    i = np.argsort(permutation)
    connectivity_reordered = exodus_a.connectivity[:, i]

    # if with_topography: Not implemented yet.

    # Find the correct weights for each point.
    nfailed = lib.triLinearInterpolator(nelem_to_search,
                                        npoints,
                                        nearest_element_indices,
                                        np.ascontiguousarray(
                                            connectivity_reordered),
                                        enclosing_elem_node_indices,
                                        np.ascontiguousarray(exodus_a.points),
                                        weights,
                                        np.ascontiguousarray(exodus_b.points))

    # interpolate the correct parameters to the new mesh.
    for param in params:
        param_a = exodus_a.get_nodal_field(param)
        values = np.sum(param_a[enclosing_elem_node_indices] * weights, axis=1)
        exodus_b.attach_field(param, np.zeros_like(values))
        exodus_b.attach_field(param, values)

    assert nfailed is 0, f"{nfailed} points could not be interpolated."


@cli.command()
@click.option('--mesh', help="Salvus continuous exodus file.", required=True)
@click.option('--gll_model', help="Salvus continuous exodus file.", required=True)
@click.option('--gll_order', help="Order of polynomials inside your gll "
                                  "model", default=4, type=int)
@click.option('--params', help="Parameters you want to interpolate to gll, "
                               "default is 'ani', you can give a list of "
                               "parameters or just give 'iso' or 'ani'",
              required=False, default=['TTI'], type=list)
# @click.option('--param', help="parameter to interpolate.", required=False)
def interpolate_mesh_to_gll(mesh, gll_model, gll_order, params=["TTI"]):
    """
    Interpolate values from normal exodus mesh to a smoothiesem gll model
    Keep this This is for 3D meshes.
    Inputs:
    :param mesh: exodus mesh file with nodal velocities.
    :param gll_model: hdf5 file to interpolate to. Polynomial order 4
    preferrably.
    :param gll_order: The order of the basis polynomials, works best for 4th
    order.
    :param params: A list of parameters to interpolate. Default: ["TTI"]
    """

    lib = load_lib()
    start = time.time()
    # Read in exodus mesh
    exodus = Exodus(mesh)
    centroids = exodus.get_element_centroid()
    centroid_tree = KDTree(centroids)

    # Read in gll model
    gll = h5py.File(gll_model, 'r+')
    if params == ["TTI"]:
        params = ["VPV", "VPH", "VSV", "VSH", "RHO", "QKAPPA", "QMU", "ETA"]

    # find coordinates in gll_model
    gll_coords = gll['MODEL']['coordinates']

    gll_points = (gll_order + 1) ** 3

    nelem_to_search = 20
    npoints = len(gll_coords)
    nearest_element_indices = np.zeros(shape=[npoints,
                                              gll_points, nelem_to_search],
                                       dtype=np.int64)


    for i in range(gll_points):
        if (i+1) % 10 == 0 or i == 124 or i == 0:
            print(f"Finding element indices for gll point: {i+1}/{gll_points}")
        _, nearest_element_indices[:, i, :] = \
            centroid_tree.query(gll_coords[:, i, :], k=nelem_to_search)

    # We need to rearrange the array to make the algorithm work
    nearest_element_indices = np.swapaxes(nearest_element_indices, 0, 1)

    enclosing_elem_node_indices = np.zeros((gll_points, npoints, 8),
                                           dtype=np.int64)
    weights = np.zeros((gll_points, npoints, 8))
    permutation = [0, 3, 2, 1, 4, 5, 6, 7]
    i = np.argsort(permutation)
    connectivity_reordered = np.ascontiguousarray(exodus.connectivity[:, i])
    exopoints = np.ascontiguousarray(exodus.points)
    nfailed = 0
    for i in range(gll_points):
        if (i+1) % 10 == 0 or i == 124 or i == 0:
            print(f"Trilinear interpolation for gll point: {i+1}/{gll_points}")
        nfailed += lib.triLinearInterpolator(nelem_to_search,
                                             npoints,
                                             np.ascontiguousarray(
                                                 nearest_element_indices[
                                                 i, :, :]),
                                             connectivity_reordered,
                                             enclosing_elem_node_indices[
                                             i, :, :],
                                             exopoints,
                                             weights[i, :, :],
                                             np.ascontiguousarray(
                                                 gll_coords[:, i, :],
                                                 dtype=np.float64))

    assert nfailed is 0, f"{nfailed} points could not be interpolated."
    # Lets just interpolate the first parameter
    # params = ["VSV", "VSH", "VPV", "VPH", "RHO"]

    isoparams = ["RHO", "VP", "VS", "QKAPPA", "QMU"]
    ttiparams = ["VPV", "VPH", "VSV", "VSH", "RHO", "ETA", "QKAPPA", "QMU"]
    # params_gll = gll["MODEL"]["data"].attrs.get("DIMENSION_LABELS")[1].decode()
    # params_gll = params_gll[2:-2].replace(" ", "").split("|")
    params_gll = isoparams
    exodusparams = ttiparams

    # if "MODEL/data" in gll:
    #     params_gll = gll["MODEL"]["data"].attrs.get("DIMENSION_LABELS")[1].decode()
    #     params_gll = params_gll[2:-2].replace(" ", "").split("|")
    #     if params_gll == params:
    del gll['MODEL/data']
    gll.create_dataset('MODEL/data', (npoints, len(isoparams), gll_points), dtype=np.float64)



    # gll['MODEL/data'].dims[0].label = 'time'
    gll['MODEL/data'].dims[0].label = 'element'
    dimstr = '[ ' + ' | '.join(isoparams) + ' ]'
    gll['MODEL/data'].dims[1].label = dimstr
    gll['MODEL/data'].dims[2].label = 'point'
    params_gll = gll["MODEL"]["data"].attrs.get("DIMENSION_LABELS")[1].decode()

    params_gll = params_gll[2:-2].replace(" ", "").split("|")
    s = 0
    # Maybe faster to just load all nodal fields to memory and use those
    for param_gll in params_gll:
        if param_gll == "VS":
            param = "VSV"
        elif param_gll == "VP":
            param = "VPV"
        else:
            param = param_gll
        param_node = exodus.get_nodal_field(param)
        for i in range(gll_points):
            if (i+1) % 10 == 0 or i == 124 or i == 0:
                print(f"Putting values onto gll points: {i+1}/{gll_points} for "
                      f"parameter {s+1}/{len(params_gll)} -- {param_gll}")

            values = np.sum(param_node[enclosing_elem_node_indices[i, :, :]] *
                            weights[i, :, :], axis=1)

            gll['MODEL']['data'][:, s, i] = values
        s += 1
    end = time.time()
    runtime = end - start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime} minutes")
    else:
        print(f"Finished in time: {runtime} seconds")


@cli.command()
@click.option('--mesh', help="Exodus file with nodal parameters.",
              required=True)
@click.option('--gll_model', help="hdf5 mesh.",
              required=True)
@click.option('--gll_order', help="Order of polynomials inside your gll "
                                  "model", default=4)
# @click.option('--params', help="parameter to interpolate.", required=False,
#               default=["TTI"], type=list)
def interpolate_gll_to_mesh(mesh, gll_model, gll_order):
    """
    A function which takes model parameters stored on GLL model and
    interpolates them on to a nodal mesh
    :param mesh: name of meshfile
    :param gll_model: name of gll_model file
    :param gll_order: order of lagrange polynomials
    """

    from multi_mesh.io.exodus import Exodus
    from pykdtree.kdtree import KDTree
    import h5py
    import sys
    import time
    start = time.time()

    # Read in gll model
    # gll = h5py.File(gll_model, 'r')
    # Compute centroids
    # if params == ["TTI"]:
    #     params = ["VPV", "VPH", "VSV", "VSH", "RHO", "QKAPPA", "QMU", "ETA"]
    # Make sure that the data labels are in correct location.
    # 'MODEL' is sometimes 'ELASTIC'
    with h5py.File(gll_model, 'r') as gll:
        gll_points = gll['MODEL']['coordinates'][:]
        gll_data = gll['MODEL']['data'][:]
        params = gll["MODEL/data"].attrs.get("DIMENSION_LABELS")[1].decode()
    centroids = _find_gll_centroids(gll_points, 3)
    print("centroids", np.shape(centroids))
    # Build a KDTree of the centroids to look for nearest elements
    print("Building KDTree")
    centroid_tree = KDTree(centroids)

    nelem_to_search = 20
    # Read in mesh
    print("Read in mesh")
    exodus = Exodus(mesh, mode="a")
    # Find nearest elements
    print("Querying the KDTree")
    _, nearest_element_indices = centroid_tree.query(exodus.points[:], k=nelem_to_search)
    npoints = exodus.npoint
    params = params[2:-2].replace(" ", "").split("|")
    print(f"Parameters to interpolate: {params_gll}")
    # sys.exit("Manual stop")
    values = np.zeros(shape=[npoints, len(params_gll)])

    # Temporary solution:
    print("nearest_element_indices", np.shape(nearest_element_indices))
    s = 0

    for point in exodus.points[:]:
        # print(f"Nearest element indices: {nearest_element_indices[s,:]}")
        element, ref_coord = _check_if_inside_element(gll_points,
                                           nearest_element_indices[s, :],
                                           point)
        coeffs = get_coefficients(4, 4, 4, ref_coord)
        func = salvus_fem.tensor_gll.GetInterpolationCoefficients
        i = 0

        for param in params:
            values[s, i] = np.sum(gll_data[element, i, :] * coeffs)
            i += 1

        if s % 20000 == 0:
            print(s)
        s += 1

    i = 0
    for param_gll in params:
        if param_gll == 'FemMassMatrix':
            continue
        if param_gll == "RHO":
            continue

        param_exod = param_gll
        exodus.attach_field(param_exod, np.zeros_like(values[:, i]))
        exodus.attach_field(param_exod, values[:, i])
        i += 1

    end = time.time()
    runtime = end-start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime} minutes")
    else:
        print(f"Finished in time: {runtime} seconds")


def get_coefficients(a, b, c, ref_coord):
    # return tensor_gll.GetInterpolationCoefficients(a, b, c, "Matrix", "Matrix", ref_coord)
    # return salvus_fem._fcts[867][1](ref_coord)
    return GetInterpolationCoefficients(ref_coord)
    # return GetInterpolationCoefficients(4, 4, 4, "Matrix", "Matrix", ref_coord)

def inverse_transform(point, gll_points):
    # return hypercube.InverseCoordinateTransformWrapper(n=4, d=3, pnt=point,
    #                                       ctrlNodes=gll_points)
    return InverseCoordinateTransformWrapper(pnt=point, ctrlNodes=gll_points)
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
        centroids[:, d] = np.mean(gll_coordinates[:, :, d], axis=1, dtype=np.float64)

    print("Found centroids")
    return centroids


def _check_if_inside_element(gll_model, nearest_elements, point):
    """
    A function to figure out inside which element the point to be interpolated is.
    :param gll: gll model
    :param nearest_elements: nearest elements of the point
    :param point: The actual point
    :return: the Index of the element which point is inside
    """
    point = np.asfortranarray(point)
    ref_coords = np.zeros(len(nearest_elements))
    l = 0
    for element in nearest_elements:
        gll_points = gll_model[element, :, :]
        gll_points = np.asfortranarray(gll_points)

        # ref_coord = salvus_fem._fcts[29][1](pnt=point, ctrlNodes=gll_points)
        ref_coord = inverse_transform(point=point, gll_points=gll_points)
        ref_coords[l] = np.sum(np.abs(ref_coord))
        l += 1

        if not np.any(np.abs(ref_coord) > 1.02):
            return element, ref_coord

    ind = np.where(ref_coords == np.min(ref_coords))[0][0]
    element = nearest_elements[ind]
    ref_coord = inverse_transform(point=point, gll_points=np.asfortranarray(
                                            gll_model[element, :, :]))
    print(f"didn't find a good fit. ref_coord: {ref_coord}")
    return element, ref_coord
    # raise IndexError("Could not find an element which this points fits into."
    #                  " Maybe you should add some tolerance")



# gll_model = "/home/solvi/workspace/InterpolationTests/smoothiesem_nlat08.h5"
# mesh = "/home/solvi/workspace/InterpolationTests/Globe3D_prem_ani_one_crust_25.e"
# gll_order = 4
# interpolate_mesh_to_gll(mesh, gll_model, 4)
