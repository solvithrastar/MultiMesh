
from multi_mesh.helpers import load_lib
from multi_mesh.io.exodus import Exodus
from multi_mesh import utils
from pykdtree.kdtree import KDTree
import h5py
import sys
import time
import numpy as np

import salvus_fem
# Buffer the salvus_fem functions, so accessing becomes much faster
for name, func in salvus_fem._fcts:
    # if name == "__GetInterpolationCoefficients__int_n0_1__int_n1_1__int_n2_0__Matrix_Derive" \
    #            "dA_Eigen::Matrix<double, 2, 1>__Matrix_DerivedB_Eigen::Matrix<double, 4, 1>":
    #     GetInterpolationCoefficients3D = func
    if name == "__GetInterpolationCoefficients__int_n0_4__int_n1_4__int_n2_4__Matrix_Derive" \
               "dA_Eigen::Matrix<double, 3, 1>__Matrix_DerivedB_Eigen::Matrix<double, 125, 1>":
        GetInterpolationCoefficients3D = func
    if name == "__InverseCoordinateTransformWrapper__int_n_4__int_d_3":
        InverseCoordinateTransformWrapper3D = func
    if name == "__GetInterpolationCoefficients__int_n0_4__int_n1_4__int_n2_0__Matrix_Derive" \
               "dA_Eigen::Matrix<double, 2, 1>__Matrix_DerivedB_Eigen::Matrix<double, 25, 1>":
        GetInterpolationCoefficients2D = func
    if name == "__InverseCoordinateTransformWrapper__int_n_4__int_d_2":
        InverseCoordinateTransformWrapper2D = func


"""
In here we have many interpolation routines. Currently there is quite a bit of
code repetition since most of this was done to solve a specific application
at the time. Hopefully I'll have time one day to make it more abstract.
"""


def exodus_2_gll(mesh, gll_model, gll_order=4, dimensions=3, nelem_to_search=20, parameters="TTI"):
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
    lib = load_lib()
    exodus, centroid_tree = utils.load_exodus(mesh, find_centroids=True)

    gll = h5py.File(gll_model, 'r+')

    gll_coords = gll['MODEL/coordinates']
    npoints = gll_coords.shape[0]
    gll_points = gll_coords.shape[1]

    nearest_element_indices = np.zeros(shape=[npoints, gll_points,
    nelem_to_search], dtype=np.int64)

    for i in range(gll_points):
        _, nearest_element_indices[:, i, :] =  centroid_tree.query(gll_coords[:, i, :], k=nelem_to_search)

    nearest_element_indices = np.swapaxes(nearest_element_indices, 0, 1)

    enclosing_elem_node_indices = np.zeros((gll_points, npoints, 8), dtype=np.int64)
    weights = np.zeros((gll_points, npoints, 8))
    permutation = [0, 3, 2, 1, 4, 5, 6, 7]
    i = np.argsort(permutation)

    # i = np.argsort(permutation)
    connectivity = np.ascontiguousarray(exodus.connectivity[:,i])
    exopoints = np.ascontiguousarray(exodus.points)
    nfailed = 0

    parameters = utils.pick_parameters(parameters)
    utils.remove_and_create_empty_dataset(gll, parameters)
    param_exodus = np.zeros(shape=(len(parameters), len(exodus.get_nodal_field(parameters[0]))))
    values = np.zeros(shape=(len(parameters), len(exodus.get_nodal_field(parameters[0]))))
    for _i, param in enumerate(parameters):
        param_exodus[_i,:] = exodus.get_nodal_field(param)

    for i in range(gll_points):
        if (i+1) % 10 == 0 or i == gll_points-1 or i == 0:
            print(f"Trilinear interpolation for gll point: {i+1}/{gll_points}")
        nfailed += lib.triLinearInterpolator(nelem_to_search,
                                             npoints,
                                             np.ascontiguousarray(
                                                 nearest_element_indices[
                                                 i, :, :], dtype=np.int64),
                                             connectivity,
                                             enclosing_elem_node_indices[
                                             i, :, :],
                                             exopoints,
                                             weights[i, :, :],
                                             np.ascontiguousarray(
                                                 gll_coords[:, i, :]))
        assert nfailed is 0, f"{nfailed} points could not be interpolated."
        values = np.sum(param_exodus[:,enclosing_elem_node_indices[i,:,:]]*weights[i,:,:], axis=2)

        gll['MODEL/data'][:,:,i] = values.T

    # s = 0
    # for i in range(gll_points):
    #     if (i+1) % 10 ==  0 or i == gll_points - 1 or i == 0:
    #         print(f"Putting values onto gll points: {i+1}/{gll_points} for "
    #               f"parameter {s+1}/{len(params_gll)} -- {param}")
    #     for param in parameters:
    #         values = np.sum(param_exodus[param][enclosing_elem_node_indices[i, :, :]] * weights[i, :, :], axis=1)
    #
    #         gll['MODEL/data'][:, s, i] = values
    # s += 1

    end = time.time()
    runtime = end - start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime} minutes")
    else:
        print(f"Finished in time: {runtime} seconds")


def gll_2_gll(from_gll, to_gll, from_gll_order=4, to_gll_order=4, dimensions=3, nelem_to_search=20, parameters="TTI"):
    """
    Interpolate parameters between two gll models.
    :param from_gll: path to gll mesh to interpolate from
    :param to_gll: path to gll mesh to interpolate to
    :param from_gll_order: order of gll_model
    :param dimensions: dimension of meshes.
    :param nelem_to_search: amount of elements to check
    :param parameters: Parameters to be interolated, possible to pass, "ISO", "TTI" or a list of parameters.
    :return: gll_mesh with new model on it
    """

    with h5py.File(from_gll, 'r') as original:
        original_points = np.array(original['MODEL/coordinates'][:], dtype=np.float64)
        original_data = original['MODEL/data'][:]
        orig_params = original["ELASTIC/data"].attrs.get("DIMENSION_LABELS")[1].decode()
        orig_params = orig_params[2:-2].replace(" ", "").replace("grad", "").split("|")

    parameters = utils.pick_parameters(parameters)
    original_centroids = _find_gll_centroids(original_points, dimension)

    original_centroid_tree = KDTree(original_centroids)

    new = h5py.File(to_gll, 'r+')

    new_points = np.array(new['MODEL/coordinates'][:], dtype=np.float64)
    if 'MODEL/data' in new:
        utils.remove_and_create_empty_dataset(new, parameters)
    # new_data = new['MODEL/data']
    # new_params = smoothie["ELASTIC/data"].attrs.get("DIMENSION_LABELS")[2].decode()
    # smoothie_params = smoothie_params[2:-2].replace(" ", "").split("|")

    map={}
    for param in parameters:
        map[param] = orig_params.index(param)

    gll_points = new['MODEL/coordinates'].shape[1]
    values = np.zeros(shape=[new_points.shape[0], len(parameters), gll_points])

    nearest_element_indices = np.zeros(shape=[new_points.shape[0],
                                              gll_points, nelem_to_search],
                                       dtype=np.int64)
    for i in range(gll_points):
        _, nearest_element_indices[:, i, :] = original_centroid_tree.query(
            new_points[:, i, :], k=nelem_to_search)

    for s in range(new_points.shape[0]):
        for i in range(gll_points):
            point = new_points[s, i, :]
            # Next two checks are only used in a current project
            # if point[0] < 0.0 or point[0] > 1.4e6:
            #     values[0, s, :, i] = smoothie_data[0, s, :, i]
            #     continue
            # if point[1] < 0.0 or point[1] > 1.4e6:
            #     values[0, s, :, i] = smoothie_data[0, s, :, i]
            #     continue
            element, ref_coord = _check_if_inside_element(
                original_points, nearest_element_indices[s, i, :], point)

            coeffs = get_coefficients(from_gll_order, to_gll_order, 4, ref_coord, dimension)
            k = 0
            for param in params:
                values[s, k, i] = np.sum(original_data[element, map[param], :] * coeffs)
                k += 1

    new['MODEL/data'] = values
    # smoothie.create_dataset('ELASTIC/data', data=values, dtype='f4')
    # smoothie['ELASTIC/data'].dims[0].label = 'time'
    # smoothie['ELASTIC/data'].dims[1].label = 'element'
    #
    # dimstr = '[ ' + ' | '.join(params) + ' ]'
    # smoothie['ELASTIC/data'].dims[2].label = dimstr
    # # smoothie['ELASTIC/new_data'] = values
    # smoothie['ELASTIC/data'].dims[3].label = 'point'


def gll_2_gll_3d(gll_a, gll_b, order):
    """
    Interpolate parameters from 2D cartesian gll mesh to a 2D smoothiesem
    gll mesh
    :param gll_a: path to mesh to interpolate from
    :param gll_b: path to mesh to interpolate to
    :param order: polynomial order of the gll basi
    """

    with h5py.File(gll_a, 'r') as gll_a:
        gll_a_points = np.array(gll_a['MODEL/coordinates'][:], dtype=np.float64)
        gll_a_data = gll_a['MODEL/data'][:]
        params = gll_a["MODEL/data"].attrs.get("DIMENSION_LABELS")[1].decode()
        params = params[2:-2].replace(" ", "").replace("grad", "").split("|")

    gll_a_centroids = _find_gll_centroids(gll_a_points, 3)
    print(gll_a_centroids.shape)

    gll_a_centroid_tree = KDTree(gll_a_centroids)

    nelem_to_search = 25
    gll_b = h5py.File(gll_b, 'r+')

    gll_b_points = np.array(gll_b['MODEL/coordinates'][:], dtype=np.float64)
    map = {}
    # if 'MODEL/data' in gll_b:
    #     gll_b_data = gll_b['MODEL/data']
    #     gll_b_params = gll_b["MODEL/data"].attrs.get("DIMENSION_LABELS")[1].decode()
    #     gll_b_params = gll_b_params[2:-2].replace(" ", "").split("|")
    #
    #     print(gll_b_params)
    #
    #     for param in params:
    #         map[param] = gll_b_params.index(param)

    # else:
    i = 0
    for param in params:
        map[param] = i
        i += 1
    gll_points = (order + 1) ** 2
    values = np.zeros(shape=[gll_b_points.shape[0], len(params), gll_points])

    nearest_element_indices = np.zeros(shape=[gll_b_points.shape[0],
                                              gll_points, nelem_to_search],
                                       dtype=np.int64)
    for i in range(gll_points):
        _, nearest_element_indices[:, i, :] = gll_a_centroid_tree.query(
            gll_b_points[:, i, :], k=nelem_to_search)
    print(f"gll_b_point.shape: {gll_b_points.shape[0]}")
    for s in range(gll_b_points.shape[0]):
        # print(f"Element: {s}")
        for i in range(gll_points):
            # print(f"gll point: {i}")
            point = gll_b_points[s, i, :]

            element, ref_coord = _check_if_inside_element(
                gll_a_points, nearest_element_indices[s, i, :], point, 3)
            # print(ref_coord)
            coeffs = get_coefficients(4, 4, 4, ref_coord, 3)
            # print(coeffs.shape)
            k = 0
            for param in params:

                values[s, map[param], i] = np.sum(gll_a_data[element, k, :] * coeffs)
                k += 1
    if 'MODEL/data' in gll_b:
        del gll_b['MODEL/data']
    gll_b.create_dataset('MODEL/data', data=values, dtype='f4')
    # gll_b['MODEL/data'].dims[0].label = 'time'
    gll_b['MODEL/data'].dims[0].label = 'element'

    dimstr = '[ ' + ' | '.join(params) + ' ]'
    gll_b['MODEL/data'].dims[1].label = dimstr
    # gll_b['MODEL/new_data'] = values
    gll_b['MODEL/data'].dims[2].label = 'point'


def gll_2_gll_gradients(simulation, master, first=True):
    """
    Interpolate gradient from simulation mesh to master model. All hdf5 format.
    This can be used to sum gradients too, by making first=False
    :param simulation: path to simulation mesh
    :param master: path to master mesh
    :param first: if false the gradient will be summed on top of existing
    gradient
    """
    with h5py.File(simulation, 'r') as sim:
        sim_points = np.array(sim['ELASTIC/coordinates'][:], dtype=np.float64)
        sim_data = sim['ELASTIC/data'][:]
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
    master = h5py.File(master, 'r+')

    master_points = np.array(master['ELASTIC/coordinates'][:], dtype=np.float64)
    master_data = master['ELASTIC/data']

    gll_points = (4 + 1) ** 2
    values = np.zeros(shape=[1, master_points.shape[0], len(params), gll_points])

    nearest_element_indices = np.zeros(shape=[master_points.shape[0],
                                              gll_points, nelem_to_search],
                                       dtype=np.int64)
    master_params = master["ELASTIC/data"].attrs.get("DIMENSION_LABELS")[2].decode()
    master_params = master_params[2:-2].replace(" ", "").replace("grad", "").split("|")
    index_map = {}
    for i in range(len(master_params)):
        if master_params[i] in params:
            index_map[i] = params.index(master_params[i])

    for i in range(gll_points):
        _, nearest_element_indices[:, i, :] = sim_centroid_tree.query(
            master_points[:, i, :], k=nelem_to_search)

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
                sim_points, nearest_element_indices[s, i, :], point)
            # print(ref_coord)
            coeffs = get_coefficients(4, 4, 0, ref_coord, 2)
            k = 0
            for param in params:
                # print(f"Parameter: {param}")
                values[0, s, k, i] = np.sum(sim_data[0, element, k, :] * coeffs)
                k += 1
    if not first:
        for key, value in index_map.items():
            values[0, :, value, :] += master_data[0, :, key, :]

    del master['ELASTIC/data']
    master.create_dataset('ELASTIC/data', data=values, dtype='f4')
    master['ELASTIC/data'].dims[0].label = 'time'
    master['ELASTIC/data'].dims[1].label = 'element'
    for i in params:
        i = "grad" + i
    dimstr = '[ ' + ' | '.join(params) + ' ]'
    master['ELASTIC/data'].dims[2].label = dimstr
    master['ELASTIC/data'].dims[3].label = 'point'


def gll_2_exodus(gll_model, exodus_model):
    """
    Interpolate parameters from gll file to exodus model. Currently I only
    need this for visualization. I could maybe make an xdmf file but that would
    be terribly boring so I'll rather do this for now.
    :param gll_model: path to gll_model
    :param exodus_model: path_to_exodus_model
    """
    with h5py.File(gll_model, 'r') as gll_model:
        gll_points = np.array(gll_model['ELASTIC/coordinates'][:], dtype=np.float64)
        gll_data = gll_model['ELASTIC/data'][:]
        params = gll_model["ELASTIC/data"].attrs.get("DIMENSION_LABELS")[2].decode()
        params = params[2:-2].replace(" ", "").split("|")

    centroids = _find_gll_centroids(gll_points, 2)
    print("centroids", np.shape(centroids))
    # Build a KDTree of the centroids to look for nearest elements
    print("Building KDTree")
    centroid_tree = KDTree(centroids)

    nelem_to_search = 20

    print("Read in mesh")
    exodus = Exodus(exodus_model, mode="a")
    # Find nearest elements
    print("Querying the KDTree")
    print(exodus.points.shape)
    if exodus.points.shape[1] == 3:
        exodus.points = exodus.points[:, :-1]
    _, nearest_element_indices = centroid_tree.query(exodus.points[:], k=nelem_to_search)
    npoints = exodus.npoint

    values = np.zeros(shape=[npoints, len(params)])

    s = 0
    for point in exodus.points[:]:
        element, ref_coord = _check_if_inside_element(gll_points,
                                                      nearest_element_indices[s, :],
                                                      point)

        coeffs = get_coefficients(4,4,0, ref_coord, 2)
        i = 0
        for param in params:
            values[s, i] = np.sum(gll_data[0, element, i, :] * coeffs)
            i += 1

        s += 1
    i = 0
    for param in params:
        exodus.attach_field(param, np.zeros_like(values[:, i]))
        exodus.attach_field(param, values[:, i])
        i += 1


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
    a_centroids = np.concatenate((a_centroids, np.zeros(
        shape=(a_centroids.shape[0], 1))), axis=1)

    centroid_tree = KDTree(a_centroids)

    nelem_to_search = 20
    exodus_b = Exodus(cartesian, mode="a")

    _, nearest_element_indices = centroid_tree.query(exodus_b.points, k=nelem_to_search)
    nearest_element_indices = np.array(nearest_element_indices, dtype=np.int64)

    npoints = exodus_b.npoint
    enclosing_element_node_indices = np.zeros((npoints, 4), dtype=np.int64)
    weights = np.zeros((npoints, 4))
    connectivity = exodus_a.connectivity[:, :]
    nfailed = lib.triLinearInterpolator(nelem_to_search,
                                        npoints,
                                        nearest_element_indices,
                                        np.ascontiguousarray(connectivity, dtype=np.int64),
                                        enclosing_element_node_indices,
                                        np.ascontiguousarray(exodus_a.points),
                                        weights,
                                        np.ascontiguousarray(exodus_b.points))

    assert nfailed is 0, f"{nfailed} points could not be interpolated"

    for param in params:
        param_a = exodus_a.get_nodal_field(param)
        values = np.sum(param_a[enclosing_element_node_indices] * weights, axis=1)
        if not first:
            param_b = exodus_b.get_nodal_field(param)  # Get pre-existing gradient
            values += param_b  # Add new gradient on top of old one
        exodus_b.attach_field(param, np.zeros_like(values))
        exodus_b.attach_field(param, values)


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

    with h5py.File(gradient, 'r') as grad:
        grad_points = np.array(grad['ELASTIC/coordinates'][:], dtype=np.float64)
        grad_data = grad['ELASTIC/data'][:]
        params = grad["ELASTIC/data"].attrs.get("DIMENSION_LABELS")[2].decode()
        params = params[2:-2].replace(" ", "").replace("grad", "").split("|")
        params = params[1:-1]

    print(params)

    grad_centroids = _find_gll_centroids(grad_points, 2)
    centroid_tree = KDTree(grad_centroids)

    nelem_to_search = 25
    cartesian = Exodus(cartesian, mode="a")

    _, nearest_element_indices = centroid_tree.query(cartesian.points[:, :2], k=nelem_to_search)
    npoints = cartesian.npoint

    values = np.zeros(shape=[npoints, len(params)])
    scaling_factor = 1.0  #34825988.0

    s = 0
    for point in cartesian.points[:, :2]:
        element, ref_coord = _check_if_inside_element(
            grad_points, nearest_element_indices[s, :], point)
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
            values[s, k] = np.sum(grad_data[0, element, k+1, :] * coeffs) * \
                           scaling_factor  # I do a k+1 because I'm not using RHO
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
    # return tensor_gll.GetInterpolationCoefficients(a, b, c, "Matrix", "Matrix", ref_coord)
    # return salvus_fem._fcts[867][1](ref_coord)
    if dimension == 3:
        return GetInterpolationCoefficients3D(ref_coord)
    elif dimension == 2:
        return GetInterpolationCoefficients2D(ref_coord)
    # return GetInterpolationCoefficients(4, 4, 4, "Matrix", "Matrix", ref_coord)


def inverse_transform(point, gll_points, dimension):
    # return hypercube.InverseCoordinateTransformWrapper(n=4, d=3, pnt=point,
    #                                       ctrlNodes=gll_points)
    if dimension == 3:
        return InverseCoordinateTransformWrapper3D(pnt=point, ctrlNodes=gll_points)
    elif dimension == 2:
        return InverseCoordinateTransformWrapper2D(pnt=point, ctrlNodes=gll_points)
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
    import warnings
    point = np.asfortranarray(point, dtype=np.float64)
    ref_coords = np.zeros(len(nearest_elements))
    l = 0
    for element in nearest_elements:
        gll_points = gll_model[element, :, :]
        gll_points = np.asfortranarray(gll_points)

        ref_coord = inverse_transform(point=point, gll_points=gll_points, dimension=dimension)
        ref_coords[l] = np.sum(np.abs(ref_coord))
        l += 1
        #salvus_fem._fcts[29][1]
        if not np.any(np.abs(ref_coord) > 1.0):
            return element, ref_coord

    warnings.warn("Could not find an element which this points fits into."
                  " Maybe you should add some tolerance."
                  " Will return the best searched element")
    ind = np.where(ref_coords == np.min(ref_coords))[0][0]
    # ind = ref_coords.index(ref_coords == np.min(ref_coords))
    element = nearest_elements[ind]

    ref_coord = inverse_transform(point=point,
                                  gll_points=np.asfortranarray(gll_model[element,:,:], dtype=np.float64),
                                  dimension=dimension)
    # element = None
    # ref_coord = None

    return element, ref_coord
