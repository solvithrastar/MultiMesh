"""
A collection of functions which perform interpolations between various meshes.
"""
import numpy as np
from multi_mesh.helpers import load_lib
from multi_mesh.io.exodus import Exodus
from multi_mesh import utils
from pykdtree.kdtree import KDTree
import h5py


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


def exodus_2_gll(mesh, gll_model, gll_order=4, dimensions=3, nelem_to_search=20, parameters="TTI", model_path="MODEL/data", coordinates_path="MODEL/coordinates"):
    """
    Interpolate parameters between exodus file and hdf5 gll file. Only works in 3 dimensions.
    :param mesh: The exodus file
    :param gll_model: The gll file
    :param gll_order: The order of the gll polynomials
    :param dimensions: How many spatial dimensions in meshes
    :param nelem_to_search: Amount of closest elements to consider
    :param parameters: Parameters to be interolated, possible to pass, "ISO", "TTI" or a list of parameters.
    """

    lib = load_lib()
    exodus, centroid_tree = utils.load_exodus(mesh, find_centroids=True)

    gll = h5py.File(gll_model, 'r+')

    gll_coords = gll[coordinates_path]
    npoints = gll_coords.shape[0]
    gll_points = gll_coords.shape[1]

    nearest_element_indices = np.zeros(shape=[npoints, gll_points,
    nelem_to_search], dtype=np.int64)

    for i in range(gll_points):
        _, nearest_element_indices[:, i, :] = centroid_tree.query(gll_coords[:, i, :], k=nelem_to_search)

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
    utils.remove_and_create_empty_dataset(gll, parameters, model_path, coordinates_path)
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
                                                 i, :, :]),
                                             connectivity,
                                             enclosing_elem_node_indices[
                                             i, :, :],
                                             exopoints,
                                             weights[i, :, :],
                                             np.ascontiguousarray(
                                                 gll_coords[:, i, :]))
        assert nfailed is 0, f"{nfailed} points could not be interpolated."
        values = np.sum(param_exodus[:,enclosing_elem_node_indices[i,:,:]]*weights[i,:,:], axis=2)

        gll[model_path][:,:,i] = values.T


def gll_2_exodus(gll_model, exodus_model, gll_order=4, dimensions=3,
                 nelem_to_search=20, parameters="TTI",
                 model_path="MODEL/data",
                 coordinates_path="MODEL/coordinates", gradient=False):
    """
    Interpolate parameters from gll file to exodus model. This will mostly be used to interpolate gradients to begin with.
    :param gll_model: path to gll_model
    :param exodus_model: path_to_exodus_model
    :param parameters: Currently not used but will be fixed later
    """
    with h5py.File(gll_model, 'r') as gll_model:
        gll_points = np.array(gll_model[coordinates_path][:], dtype=np.float64)
        gll_data = gll_model[model_path][:]
        params = gll_model[model_path].attrs.get("DIMENSION_LABELS")[1].decode()
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
    _, nearest_element_indices = centroid_tree.query(exodus.points, k=nelem_to_search)
    npoints = exodus.npoint
    # parameters = utils.pick_parameters(parameters)
    values = np.zeros(shape=[npoints, len(parameters)])
    print(parameters)
    s = 0

    for point in exodus.points:
        if s == 0 or (s+1) % 1000 == 0:
            print(f"Now I'm looking at point number: {s+1}/{len(exodus.points)}")
        element, ref_coord = _check_if_inside_element(gll_points,
                                                      nearest_element_indices[s, :],
                                                      point, dimensions)

        coeffs = get_coefficients(4,4,0, ref_coord, dimensions)
        values[s, :] = np.sum(gll_data[element, :, :] * coeffs, axis=1)
        s += 1
    i = 0
    for param in parameters:
        exodus.attach_field(param, np.zeros_like(values[:, i]))
        exodus.attach_field(param, values[:, i])
        i += 1


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


gll_2_exodus("/home/solvi/workspace/InterpolationTests/multi_mesh_test/gradient.h5",
             "/home/solvi/workspace/InterpolationTests/multi_mesh_test/Globe3D_csem_70.e",
             gll_order=4, dimensions=3,
                 nelem_to_search=20, parameters="TTI",
                 model_path="MODEL/data",
                 coordinates_path="MODEL/coordinates", gradient=False)