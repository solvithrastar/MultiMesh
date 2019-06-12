"""
A collection of functions which perform interpolations between various meshes.
"""
import numpy as np
from multi_mesh.helpers import load_lib
from multi_mesh.io.exodus import Exodus
from multi_mesh import utils
from pykdtree.kdtree import KDTree
import h5py

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
