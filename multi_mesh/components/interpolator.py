"""
A collection of functions which perform interpolations between various meshes.
"""
import numpy as np
# from multi_mesh.helpers import load_lib
# from multi_mesh.io.exodus import Exodus
from multi_mesh import utils
from pykdtree.kdtree import KDTree
import h5py


import salvus_fem
# Buffer the salvus_fem functions, so accessing becomes much faster
for name, func in salvus_fem._fcts:
    if name == "__GetInterpolationCoefficients__int_n0_4__int_n1_4__int_n2_4__Matrix_Derive" \
               "dA_Eigen::Matrix<double, 3, 1>__Matrix_DerivedB_Eigen::Matrix<double, 125, 1>":
        GetInterpolationCoefficients3D_order_4 = func
    if name == "__GetInterpolationCoefficients__int_n0_2__int_n1_2__int_n2_2__Matrix_Derive" \
               "dA_Eigen::Matrix<double, 3, 1>__Matrix_DerivedB_Eigen::Matrix<double, 27, 1>":
        GetInterpolationCoefficients3D_order_2 = func
    if name == "__InverseCoordinateTransformWrapper__int_n_4__int_d_3":
        InverseCoordinateTransformWrapper3D_4 = func
    if name == "__InverseCoordinateTransformWrapper__int_n_2__int_d_3":
        InverseCoordinateTransformWrapper3D_2 = func
    if name == "__GetInterpolationCoefficients__int_n0_4__int_n1_4__int_n2_0__Matrix_Derive" \
               "dA_Eigen::Matrix<double, 2, 1>__Matrix_DerivedB_Eigen::Matrix<double, 25, 1>":
        GetInterpolationCoefficients2D = func
    if name == "__InverseCoordinateTransformWrapper__int_n_4__int_d_2":
        InverseCoordinateTransformWrapper2D = func
    if name == "__CheckHullWrapper__int_n_4__int_d_3":
        CheckHull = func


def exodus_2_gll(mesh, gll_model, gll_order=4, dimensions=3,
                 nelem_to_search=20, parameters="TTI",
                 model_path="MODEL/data",
                 coordinates_path="MODEL/coordinates"):
    """
    Interpolate parameters between exodus file and hdf5 gll file.
    Only works in 3 dimensions.
    :param mesh: The exodus file
    :param gll_model: The gll file
    :param gll_order: The order of the gll polynomials
    :param dimensions: How many spatial dimensions in meshes
    :param nelem_to_search: Amount of closest elements to consider
    :param parameters: Parameters to be interolated, possible to pass, "ISO",
    "TTI" or a list of parameters.
    """

    lib = load_lib()
    exodus, centroid_tree = utils.load_exodus(mesh, find_centroids=True)

    gll = h5py.File(gll_model, 'r+')

    gll_coords = gll[coordinates_path]
    npoints = gll_coords.shape[0]
    gll_points = gll_coords.shape[1]

    nearest_element_indices = np.zeros(shape=[npoints, gll_points,
                                              nelem_to_search],
                                       dtype=np.int64)

    for i in range(gll_points):
        _, nearest_element_indices[:, i, :] = centroid_tree.query(
            gll_coords[:, i, :], k=nelem_to_search)

    nearest_element_indices = np.swapaxes(nearest_element_indices, 0, 1)

    enclosing_elem_node_indices = np.zeros((gll_points, npoints, 8),
                                           dtype=np.int64)
    weights = np.zeros((gll_points, npoints, 8))
    permutation = [0, 3, 2, 1, 4, 5, 6, 7]
    i = np.argsort(permutation)

    # i = np.argsort(permutation)
    connectivity = np.ascontiguousarray(exodus.connectivity[:, i])
    exopoints = np.ascontiguousarray(exodus.points)
    nfailed = 0

    parameters = utils.pick_parameters(parameters)
    utils.remove_and_create_empty_dataset(gll, parameters, model_path,
                                          coordinates_path)
    param_exodus = np.zeros(shape=(len(parameters),
                            len(exodus.get_nodal_field(parameters[0]))))
    values = np.zeros(shape=(len(parameters),
                             len(exodus.get_nodal_field(parameters[0]))))
    for _i, param in enumerate(parameters):
        param_exodus[_i, :] = exodus.get_nodal_field(param)

    for i in range(gll_points):
        if (i+1) % 10 == 0 or i == gll_points-1 or i == 0:
            print(f"Trilinear interpolation for gll point: {i+1}/{gll_points}")
        nfailed += lib.triLinearInterpolator(nelem_to_search,
                                             npoints,
                                             np.ascontiguousarray(
                                                 nearest_element_indices[
                                                    i, :, :]),
                                             connectivity,
                                             enclosing_elem_node_indices[
                                                    i, :, :],
                                             exopoints,
                                             weights[i, :, :],
                                             np.ascontiguousarray(
                                                 gll_coords[:, i, :]))
        assert nfailed is 0, f"{nfailed} points could not be interpolated."
        values = np.sum(param_exodus[:,
                        enclosing_elem_node_indices[i, :, :]] * weights[
                        i, :, :], axis=2)

        gll[model_path][:, :, i] = values.T


def gll_2_exodus(gll_model, exodus_model, gll_order=4, dimensions=3,
                 nelem_to_search=20, parameters="TTI",
                 model_path="MODEL/data",
                 coordinates_path="MODEL/coordinates", gradient=False):
    """
    Interpolate parameters from gll file to exodus model. This will mostly be
    used to interpolate gradients to begin with.
    :param gll_model: path to gll_model
    :param exodus_model: path_to_exodus_model
    :param parameters: Currently not used but will be fixed later
    """
    with h5py.File(gll_model, 'r') as gll_model:
        gll_points = np.array(gll_model[coordinates_path][:], dtype=np.float64)
        gll_data = gll_model[model_path][:]
        params = gll_model[model_path].attrs.get(
                    "DIMENSION_LABELS")[1].decode()
        parameters = params[2:-2].replace(" ", "").split("|")

    centroids = _find_gll_centroids(gll_points, dimensions)
    print("centroids", np.shape(centroids))
    # Build a KDTree of the centroids to look for nearest elements
    print("Building KDTree")
    centroid_tree = KDTree(centroids)

    print("Read in mesh")
    exodus = Exodus(exodus_model, mode="a")
    # Find nearest elements
    print("Querying the KDTree")
    print(exodus.points.shape)
    # if exodus.points.shape[1] == 3:
    #     exodus.points = exodus.points[:, :-1]
    _, nearest_element_indices = centroid_tree.query(exodus.points,
                                                     k=nelem_to_search)
    npoints = exodus.npoint
    # parameters = utils.pick_parameters(parameters)
    values = np.zeros(shape=[npoints, len(parameters)])
    print(parameters)
    s = 0

    for point in exodus.points:
        if s == 0 or (s+1) % 1000 == 0:
            print(f"Now I'm looking at point number:"
                  f"{s+1}{len(exodus.points)}")
        element, ref_coord = _check_if_inside_element(gll_points,
                                                      nearest_element_indices[
                                                       s, :],
                                                      point, dimensions)

        coeffs = get_coefficients(4, 4, 0, ref_coord, dimensions)
        values[s, :] = np.sum(gll_data[element, :, :] * coeffs, axis=1)
        s += 1
    i = 0
    for param in parameters:
        exodus.attach_field(param, np.zeros_like(values[:, i]))
        exodus.attach_field(param, values[:, i])
        i += 1

def gll_2_gll(from_gll, to_gll, from_gll_order=4, to_gll_order=4, dimensions=3,
              nelem_to_search=20, parameters="ISO", from_model_path="MODEL/data",
              to_model_path="MODEL/data", from_coordinates_path="MODEL/coordinates",
              to_coordinates_path="MODEL/coordinates"):
    """
    Interpolate parameters between two gll models.
    It loads from_gll to memory, looks at the points of the to_gll and
    assembles a list of unique points which it interpolates onto.
    It then reconstructs the point values based on the initial to_gll points
    and saves it to file.

    :param from_gll: path to gll mesh to interpolate from
    :param to_gll: path to gll mesh to interpolate to
    :param from_gll_order: order of gll_model
    :param dimensions: dimension of meshes.
    :param nelem_to_search: amount of elements to check
    :param parameters: Parameters to be interpolated, possible to pass, "ISO", "TTI" or a list of parameters.
    """
    from tqdm import tqdm
    print("Initialization stage")
    original_points, original_data, original_params = utils.load_hdf5_params_to_memory(
        from_gll, from_model_path, from_coordinates_path)

    parameters = utils.pick_parameters(parameters)
    assert set(parameters) <= set(
        original_params), f"Original mesh does not have all the parameters you wish to interpolate. You asked for {parameters}, mesh has {original_params}"

    all_old_points = original_points.reshape(original_points.shape[0] * original_points.shape[1], original_points.shape[2])
    original_tree = KDTree(all_old_points)
    new = h5py.File(to_gll, 'r+')

    new_points = np.array(new[to_coordinates_path][:], dtype=np.float64)

    permutation = np.arange(0, len(parameters))
    for _i, param in enumerate(original_params):
        permutation[_i] = parameters.index(param)

    # In case we are not interpolating all the parameters.
    if len(permutation) != len(original_params):
        for i in range(len(original_params) - len(permutation)):
            permutation.append(np.max(permutation) + 1)

    # Check if there is some need for reordering of parameters.
    reorder = False
    for i in range(len(permutation)):
        if i == 0:
            if permutation[i] != 0:
                reorder = True
                break
        else:
            if permutation[i] != permutation[i-1] + 1:
                reorder = True
                break

    if reorder:
        args = np.argsort(permutation).astype(int)
    else:
        args = np.arange(start=0, stop=len(permutation)).astype(int)
    parameters = [parameters[x] for x in args]

    gll_points = new[to_coordinates_path].shape[1]

    all_new_points = new_points.reshape((new_points.shape[0]*new_points.shape[1], new_points.shape[2]))
    unique_new_points, recon = np.unique(all_new_points, return_inverse=True, axis=0)
 
    nearest_element_indices = np.zeros(shape=[unique_new_points.shape[0], nelem_to_search], dtype=np.int)

    _, nearest_element_indices[:, :] = original_tree.query(unique_new_points[:, :], k=nelem_to_search)
    nearest_element_indices = np.floor(nearest_element_indices/original_points.shape[1]).astype(int)

    # 
    nearest_element_indices = np.swapaxes(nearest_element_indices, 0, 1)
    unique_new_points = np.swapaxes(unique_new_points, 0, 1)
    coeffs = np.zeros(shape=[len(parameters), original_points.shape[1], unique_new_points.shape[1]])

    element = np.zeros(shape=unique_new_points.shape[1])
    print("Now we start interpolating")
    for i in tqdm(range(unique_new_points.shape[1])):
        # if i % 10000 == 0:
        #     print(f"Interpolating point number {i}/{unique_new_points.shape[1]}")
        element[i], ref_coord = _check_if_inside_element(original_points, nearest_element_indices[:, i], unique_new_points[:,i], dimensions)
        coeffs[0, :, i] = get_coefficients(from_gll_order, from_gll_order, from_gll_order, ref_coord, dimensions)

    for i in range(len(parameters))[1:]:
        coeffs[i, :, :] = coeffs[0, :, :]
    print("Interpolation done, Need to organize the results and write to file")
    element = element.astype(int)

    resample_data = original_data[element]

    # values = np.zeros(shape=[len(unique_new_points), len(parameters), gll_points], dtype=np.float64)
    coeffs = np.swapaxes(coeffs, 0, 2)
    coeffs = np.swapaxes(coeffs, 1, 2)
    values = np.sum(resample_data[:, :, :] * coeffs[:, :, :], axis=2)
    values = values[recon, :]
    values = values.reshape((new_points.shape[0], gll_points, len(args)))
    values = np.swapaxes(values, 1, 2)
    utils.remove_and_create_empty_dataset(new, parameters, to_model_path,
                                          to_coordinates_path)

    new[to_model_path][:, :, :] = values


def linear_gll_2_ex(gll_model, exodus_model, gll_order=4, dimensions=3,
                    nelem_to_search=20, parameters="TTI",
                    model_path="MODEL/data",
                    coordinates_path="MODEL/coordinates", gradient=False):
    """
    Interpolate parameters from gll file to exodus model. This will mostly be
    used to interpolate gradients to begin with.
    :param gll_model: path to gll_model
    :param exodus_model: path_to_exodus_model
    :param parameters: Currently not used but will be fixed later
    """
    with h5py.File(gll_model, 'r') as gll_model:
        gll_points = np.array(gll_model[coordinates_path][:], dtype=np.float64)
        gll_data = gll_model[model_path][:]
        params = gll_model[model_path].attrs.get(
                    "DIMENSION_LABELS")[1].decode()
        parameters = params[2:-2].replace(" ", "").split("|")

    centroids = _find_gll_centroids(gll_points, dimensions)
    print("centroids", np.shape(centroids))
    # Build a KDTree of the centroids to look for nearest elements
    print("Building KDTree")
    centroid_tree = KDTree(centroids)

    nelem_to_search = 2

    print("Read in mesh")
    exodus = Exodus(exodus_model, mode="a")
    # Find nearest elements
    print("Querying the KDTree")
    print(exodus.points.shape)
    # if exodus.points.shape[1] == 3:
    #     exodus.points = exodus.points[:, :-1]
    _, nearest_element_indices = centroid_tree.query(exodus.points,
                                                     k=nelem_to_search)
    npoints = exodus.npoint
    # parameters = utils.pick_parameters(parameters)
    values = np.zeros(shape=[npoints, len(parameters)])
    print(parameters)
    s = 0

    for i in range(gll_points):
        if (i+1) % 10 == 0 or i == gll_points-1 or i == 0:
            print(f"Trilinear interpolation for gll point: {i+1}/{gll_points}")
        nfailed += lib.triLinearInterpolator(nelem_to_search,
                                             npoints,
                                             np.ascontiguousarray(
                                                 nearest_element_indices[
                                                    i, :, :]),
                                             connectivity,
                                             enclosing_elem_node_indices[
                                                    i, :, :],
                                             exopoints,
                                             weights[i, :, :],
                                             np.ascontiguousarray(
                                                 gll_coords[:, i, :]))
        assert nfailed is 0, f"{nfailed} points could not be interpolated."
        values = np.sum(param_exodus[:,
                        enclosing_elem_node_indices[i, :, :]] * weights[
                        i, :, :], axis=2)

        gll[model_path][:, :, i] = values.T


    for point in exodus.points:
        if s == 0 or (s+1) % 1000 == 0:
            print(f"Now I'm looking at point number:"
                  f"{s+1}{len(exodus.points)}")
        element, ref_coord = _check_if_inside_element(gll_points,
                                                      nearest_element_indices[
                                                       s, :],
                                                      point, dimensions)

        coeffs = get_coefficients(4, 4, 0, ref_coord, dimensions)
        values[s, :] = np.sum(gll_data[element, :, :] * coeffs, axis=1)
        s += 1
    i = 0
    for param in parameters:
        exodus.attach_field(param, np.zeros_like(values[:, i]))
        exodus.attach_field(param, values[:, i])
        i += 1


def get_coefficients(a, b, c, ref_coord, dimension):

    if dimension == 3:
        if a == 4:
            return GetInterpolationCoefficients3D_order_4(ref_coord)
        elif a == 2:
            return GetInterpolationCoefficients3D_order_2(ref_coord)
    elif dimension == 2:
        return GetInterpolationCoefficients2D(ref_coord)

def boundary_box_check(point, gll_points) -> bool:
    """
    Check whether point is within the boundary of the box around 
    the element.
    
    :param point: Point to investigate
    :type point: numpy array
    :param gll_points: Control nodes of the element
    :type gll_points: numpy array
    """
    p_min, p_max = gll_points.min(axis=0), gll_points.max(axis=0)
    dist = 0
    if (point >= p_min).all() and (point <= p_max).all():
        return True, dist
    else:
        center = np.mean(gll_points, axis=0)
        dist += np.linalg.norm(point - center)
        return False, dist
        
        



def inverse_transform(point, gll_points, dimension):
    # return hypercube.InverseCoordinateTransformWrapper(n=4, d=3, pnt=point,
    #                                       ctrlNodes=gll_points)
    if dimension == 3:
        if len(gll_points) == 125:
            return InverseCoordinateTransformWrapper3D_4(pnt=point, ctrlNodes=gll_points)
        if len(gll_points) == 27:
            return InverseCoordinateTransformWrapper3D_2(pnt=point, ctrlNodes=gll_points)
    elif dimension == 2:
        return InverseCoordinateTransformWrapper2D(pnt=point,
                                                   ctrlNodes=gll_points)
    # return salvus_fem._fcts[29][1](pnt=point, ctrlNodes=gll_points)


def _find_gll_centroids(gll_coordinates, dimensions=3):
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
        centroids[:, d] = np.mean(gll_coordinates[:, :, d], axis=1,
                                  dtype=np.float64)

    # print("Found centroids")
    return centroids


def _check_if_inside_element(gll_model, nearest_elements, point, dimension):
    """
    A function to figure out inside which element the point to be interpolated
    is.
    :param gll: gll model
    :param nearest_elements: nearest elements of the point
    :param point: The actual point
    :return: the Index of the element which point is inside
    """
    import warnings
    point = np.asfortranarray(point, dtype=np.float64)
    dist = np.zeros(len(nearest_elements))
    for _i, element in enumerate(nearest_elements):
        gll_points = gll_model[element, :, :]
        gll_points = np.asfortranarray(gll_points)
        inside, dist[_i] = boundary_box_check(point, gll_points)
        if inside:
            # I can implement this to save best results
            ref_coord = inverse_transform(point=point, gll_points=gll_points,
                                          dimension=dimension)
            return element, ref_coord
        
        # ref_coords[_i] = np.sum(np.abs(ref_coord))
        # if CheckHull(ref_coord, gll_points):
        #     return element, ref_coord
        # if not np.any(np.abs(ref_coord) > 1.02):

        #     return element, ref_coord

    warnings.warn("Could not find an element which this points fits into."
                  " Maybe you should add some tolerance."
                  " Will return the best searched element")
    ind = np.where(dist == np.min(dist))[0][0]
    # # ind = ref_coords.index(ref_coords == np.min(ref_coords))
    element = nearest_elements[ind]
    ref_coord = inverse_transform(point=point, gll_points=
                                  np.asfortranarray(gll_model[element, :, :],
                                                    dtype=np.float64),
                                  dimension=dimension)
    # # element = None
    # # ref_coord = None

    return element, ref_coord


# gll_2_exodus("/home/solvi/workspace/InterpolationTests/multi_mesh_test/gradient.h5",
#              "/home/solvi/workspace/InterpolationTests/multi_mesh_test/Globe3D_csem_70.e",
#              gll_order=4, dimensions=3,
#                  nelem_to_search=20, parameters="TTI",
#                  model_path="MODEL/data",
#                  coordinates_path="MODEL/coordinates", gradient=False)
