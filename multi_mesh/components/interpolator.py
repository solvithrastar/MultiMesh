"""
A collection of functions which perform interpolations between various meshes.
"""
import numpy as np

# from multi_mesh.helpers import load_lib
# from multi_mesh.io.exodus import Exodus
from multi_mesh import utils
from pykdtree.kdtree import KDTree
import h5py
import sys
import salvus.fem
import os
from tqdm import tqdm
from typing import Dict, List, Union, Tuple
from collections import defaultdict
import multiprocessing
from multi_mesh.components.salvus_mesh_reader import SalvusMesh
import pathlib


# Buffer the salvus_fem functions, so accessing becomes much faster
for name, func in salvus.fem._fcts:
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
    if (
        name
        == "__GetInterpolationCoefficients__int_n0_1__int_n1_1__int_n2_1__Matrix_Derive"
        "dA_Eigen::Matrix<double, 3, 1>__Matrix_DerivedB_Eigen::Matrix<double, 8, 1>"
    ):
        GetInterpolationCoefficients3D_order_1 = func
    if name == "__InverseCoordinateTransformWrapper__int_n_4__int_d_3":
        InverseCoordinateTransformWrapper3D_4 = func
    if name == "__InverseCoordinateTransformWrapper__int_n_2__int_d_3":
        InverseCoordinateTransformWrapper3D_2 = func
    if name == "__InverseCoordinateTransformWrapper__int_n_1__int_d_3":
        InverseCoordinateTransformWrapper3D_1 = func
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


def query_model(
    coordinates, model, nelem_to_search, model_path, coordinates_path,
):
    """
    Provide an array of coordinates, returns an array with model parameters
    for each of these coordinates.

    :param coordinates: Array of coordinates in lat lon depth_in_m
    :type coordinates: np.array, dimensions N,3
    :param model: Salvus mesh with model stored on it
    :type model: hdf5 salvus mesh file
    :param nelem_to_search: Number of elements to KDtree query
    :type nelem_to_search: int
    :param model_path: Where are parameters stored?
    :type model_path: str
    :param coordinates_path: Where are coordinates stored?
    :type coordinates_path: str
    :return: An array of parameters
    :rtype: np.array
    """
    from multi_mesh.utils import latlondepth_to_xyz

    print("Initialization stage")
    (
        original_points,
        original_data,
        original_params,
    ) = utils.load_hdf5_params_to_memory(model, model_path, coordinates_path)

    dimensions = original_points.shape[2]
    gll_order = int(round(original_data.shape[2] ** (1.0 / dimensions))) - 1

    # Reshape points to remove the dimension of the gll points
    all_original_points = original_points.reshape(
        original_points.shape[0] * original_points.shape[1],
        original_points.shape[2],
    )
    original_tree = KDTree(all_original_points)

    assert (
        coordinates.shape[1] == 3
    ), "Make sure coordinates array has shape N,3"
    coordinates = latlondepth_to_xyz(latlondepth=coordinates)
    _, nearest_element_indices = original_tree.query(
        coordinates, k=nelem_to_search
    )
    # We need to get the arrays ready for the interpolation function
    nearest_element_indices = np.swapaxes(nearest_element_indices, 0, 1)
    coordinates = np.swapaxes(coordinates, 0, 1)
    coeffs_empty = np.zeros(
        shape=(
            len(original_params),
            original_points.shape[1],
            coordinates.shape[1],
        )
    )
    nearest_element_indices = np.floor(
        nearest_element_indices / original_points.shape[1]
    ).astype(int)
    element_empty = np.zeros(shape=coordinates.shape[1])
    elements, coeffs = find_gll_coeffs(
        original_coordinates=original_points,
        coordinates=coordinates,
        nearest_elements=nearest_element_indices,
        coeffs=coeffs_empty,
        element=element_empty,
        dimensions=dimensions,
        from_gll_order=gll_order,
        ignore_hard_elements=False,
    )

    # return elements, coeffs
    for i in range(len(original_params))[1:]:
        coeffs[i, :, :] = coeffs[0, :, :]
    print("Interpolation done, need to organize the results")
    elements = elements.astype(int)
    coeffs = np.swapaxes(coeffs, 0, 2).swapaxes(1, 2)
    resample_data = original_data[elements]
    values = np.sum(resample_data[:, :, :] * coeffs[:, :, :], axis=2)[:, :]
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
    :param parameters: Parameters to be interolated, possible to pass, "ISO",
    "TTI" or a list of parameters.
    """

    lib = load_lib()
    exodus, centroid_tree = utils.load_exodus(mesh, find_centroids=True)

    gll = h5py.File(gll_model, "r+")

    gll_coords = gll[coordinates_path]
    npoints = gll_coords.shape[0]
    gll_points = gll_coords.shape[1]

    nearest_element_indices = np.zeros(
        shape=[npoints, gll_points, nelem_to_search], dtype=np.int64
    )

    for i in range(gll_points):
        _, nearest_element_indices[:, i, :] = centroid_tree.query(
            gll_coords[:, i, :], k=nelem_to_search
        )

    nearest_element_indices = np.swapaxes(nearest_element_indices, 0, 1)

    enclosing_elem_node_indices = np.zeros(
        (gll_points, npoints, 8), dtype=np.int64
    )
    weights = np.zeros((gll_points, npoints, 8))
    permutation = [0, 3, 2, 1, 4, 5, 6, 7]
    i = np.argsort(permutation)

    # i = np.argsort(permutation)
    connectivity = np.ascontiguousarray(exodus.connectivity[:, i])
    exopoints = np.ascontiguousarray(exodus.points)
    nfailed = 0

    parameters = utils.pick_parameters(parameters)
    utils.remove_and_create_empty_dataset(
        gll, parameters, model_path, coordinates_path
    )
    param_exodus = np.zeros(
        shape=(len(parameters), len(exodus.get_nodal_field(parameters[0])))
    )
    values = np.zeros(
        shape=(len(parameters), len(exodus.get_nodal_field(parameters[0])))
    )
    for _i, param in enumerate(parameters):
        param_exodus[_i, :] = exodus.get_nodal_field(param)

    for i in range(gll_points):
        if (i + 1) % 10 == 0 or i == gll_points - 1 or i == 0:
            print(f"Trilinear interpolation for gll point: {i+1}/{gll_points}")
        nfailed += lib.triLinearInterpolator(
            nelem_to_search,
            npoints,
            np.ascontiguousarray(nearest_element_indices[i, :, :]),
            connectivity,
            enclosing_elem_node_indices[i, :, :],
            exopoints,
            weights[i, :, :],
            np.ascontiguousarray(gll_coords[:, i, :]),
        )
        assert nfailed is 0, f"{nfailed} points could not be interpolated."
        values = np.sum(
            param_exodus[:, enclosing_elem_node_indices[i, :, :]]
            * weights[i, :, :],
            axis=2,
        )

        gll[model_path][:, :, i] = values.T


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
    Interpolate parameters from gll file to exodus model. This will mostly be
    used to interpolate gradients to begin with.
    :param gll_model: path to gll_model
    :param exodus_model: path_to_exodus_model
    :param parameters: Currently not used but will be fixed later
    """
    with h5py.File(gll_model, "r") as gll_model:
        gll_points = np.array(gll_model[coordinates_path][:], dtype=np.float64)
        gll_data = gll_model[model_path][:]
        params = (
            gll_model[model_path].attrs.get("DIMENSION_LABELS")[1].decode()
        )
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
    _, nearest_element_indices = centroid_tree.query(
        exodus.points, k=nelem_to_search
    )
    npoints = exodus.npoint
    # parameters = utils.pick_parameters(parameters)
    values = np.zeros(shape=[npoints, len(parameters)])
    print(parameters)
    s = 0

    for point in exodus.points:
        if s == 0 or (s + 1) % 1000 == 0:
            print(
                f"Now I'm looking at point number:"
                f"{s+1}{len(exodus.points)}"
            )
        element, ref_coord = _check_if_inside_element(
            gll_points, nearest_element_indices[s, :], point, dimensions
        )

        coeffs = get_coefficients(4, 4, 0, ref_coord, dimensions)
        values[s, :] = np.sum(gll_data[element, :, :] * coeffs, axis=1)
        s += 1
    i = 0
    for param in parameters:
        exodus.attach_field(param, np.zeros_like(values[:, i]))
        exodus.attach_field(param, values[:, i])
        i += 1


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

    print("Initialization stage")
    print(f"Stored array: {stored_array}")
    original_mesh = SalvusMesh(from_gll, fast_mode=False)
    if make_spherical:
        map_to_sphere(original_mesh)
    original_mask, layers = utils.create_layer_mask(
        mesh=original_mesh, layers=layers
    )
    if parameters == "all":
        parameters = list(original_mesh.element_nodal_fields.keys())
    # original_points = original_mesh.get_element_nodes()[original_mask]

    dimensions = 3

    new_mesh = SalvusMesh(to_gll, fast_mode=False)
    if make_spherical:
        map_to_sphere(new_mesh)
    # Stored array stuff here
    loop = True
    if stored_array is not None and os.path.exists(
        os.path.join(stored_array, "interp_info.h5")
    ):
        print("No need for looping, we have the matrices")
        loop = False
        dataset = h5py.File(os.path.join(stored_array, "interp_info.h5"), "r")
        coeffs = dataset["coeffs"]
        elements = dataset["elements"]

    # Unique new points is a dictionary with tuples of coordinates and a
    # reconstruction array
    unique_new_points, mask, layers = utils.get_unique_points(
        points=new_mesh, mesh=True, layers=layers
    )
    parameters = utils.pick_parameters(parameters)

    original_trees = {}
    nearest_element_indices = {}
    # Making a dictionary of KDTrees, one per layer
    # Now we do it based on element centroids
    if loop:
        for layer in layers:
            layer = str(layer)
            points = original_mesh.get_element_centroids()[
                original_mask[layer]
            ]
            original_trees[layer] = KDTree(points)
            nearest_element_indices[layer] = np.zeros(
                shape=(unique_new_points[layer][0].shape[0], nelem_to_search),
                dtype=np.int,
            )
            _, nearest_element_indices[layer][:, :] = original_trees[
                layer
            ].query(unique_new_points[layer][0], k=nelem_to_search)

        # I should try the tri-linear interpolation here too. But then I KDTree to the gll points.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

        from_gll_order = original_mesh.shape_order
        # Time consuming part
        coeffs, elements = fill_value_array(
            new_coordinates=unique_new_points,
            nearest_elements=nearest_element_indices,
            original_mesh=original_mesh,
            original_mask=original_mask,
            parameters=parameters,
            dimensions=dimensions,
            from_gll_order=from_gll_order,
        )
        num_failed = 0

        if stored_array is not None:
            print("Saving interpolation matrices")
            dataset = h5py.File(
                os.path.join(stored_array, "interp_info.h5"), "w"
            )
            for k, v in coeffs.items():
                dataset.create_dataset(f"coeffs/{k}", data=v)
            for k, v in elements.items():
                dataset.create_dataset(f"elements/{k}", data=v)
            dataset.close()

    for layer in coeffs.keys():
        if loop:
            num_failed += len(np.where(elements[layer] == -1)[0])
            elms = elements[layer].astype(int)
        else:
            elms = elements[layer][()].astype(int)
        for param in parameters:
            values = np.sum(
                original_mesh.element_nodal_fields[param][
                    original_mask[layer]
                ][elms]
                * coeffs[layer],
                axis=1,
            )
            new_mesh.element_nodal_fields[param][mask[layer]] = values[
                unique_new_points[layer][1]
            ].reshape(new_mesh.element_nodal_fields[param][mask[layer]].shape)
    for _i, param in enumerate(parameters):
        new_field = np.zeros_like(new_mesh.element_nodal_fields[param])
        for layer in coeffs.keys():
            elms = elements[layer].astype(int)
            values = np.sum(
                original_mesh.element_nodal_fields[param][
                    original_mask[layer]
                ][elms]
                * coeffs[layer],
                axis=1,
            )
            new_field[mask[layer]] = values[
                unique_new_points[layer][1]
            ].reshape(new_mesh.element_nodal_fields[param][mask[layer]].shape)
        new_mesh.attach_field(name=param, data=new_field)

    # for param in parameters:
    #     values = np.sum(
    #         original_mesh.element_nodal_fields[param][original_mask][elements]
    #         * coeffs,
    #         axis=1,
    #     )
    #     new_mesh.element_nodal_fields[param][mask] = values[recon].reshape(
    #         new_mesh.element_nodal_fields[param][mask].shape
    #     )

    # new_mesh.write_h5(to_gll)


def gll_2_gll_layered_multi(
    from_gll: Union[str, pathlib.Path],
    to_gll: Union[str, pathlib.Path],
    layers: Union[List[int], str],
    nelem_to_search: int = 20,
    parameters: Union[List[str], str] = "all",
    threads: int = None,
    stored_array: Union[str, pathlib.Path] = None,
    make_spherical: bool = False,
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

    # from salvus.mesh.unstructured_mesh import UnstructuredMesh
    from multi_mesh.components.salvus_mesh_reader import SalvusMesh

    manager = multiprocessing.Manager()
    elements = manager.dict()
    coeffs = manager.dict()

    print("Initialization stage")
    # print(f"Stored array: {stored_array}")
    original_mesh = SalvusMesh(from_gll, fast_mode=False)
    if make_spherical:
        map_to_sphere(original_mesh)
    original_mask, layers = utils.create_layer_mask(
        mesh=original_mesh, layers=layers
    )
    if parameters == "all":
        parameters = list(original_mesh.element_nodal_fields.keys())
    dimensions = 3
    new_mesh = SalvusMesh(to_gll, fast_mode=False)
    if make_spherical:
        map_to_sphere(new_mesh)

    loop = True
    if stored_array is not None and os.path.exists(
        os.path.join(stored_array, "interp_info.h5")
    ):
        print("No need for looping, we have the matrices")
        loop = False
        dataset = h5py.File(os.path.join(stored_array, "interp_info.h5"), "r")
        coeffs = dataset["coeffs"]
        elements = dataset["elements"]
    unique_new_points, mask, layers = utils.get_unique_points(
        points=new_mesh, mesh=True, layers=layers
    )
    parameters = utils.pick_parameters(parameters)

    global _find_interpolation_weights

    def _find_interpolation_weights(layer):
        # Find the interpolation weights for this particular layer.
        # TODO: Try doing the KDTree before
        points = original_mesh.get_element_centroids()[original_mask[layer]]
        original_tree = KDTree(points)
        nearest_element_indices = np.zeros(
            shape=unique_new_points[layer][0].shape[0], dtype=np.int
        )
        _, nearest_element_indices = original_tree.query(
            unique_new_points[layer][0], k=nelem_to_search
        )

        from_gll_order = original_mesh.shape_order

        def _fill_value_array(
            new_coordinates,
            nearest_elements,
            original_mesh,
            original_mask,
            parameters,
            dimensions,
            from_gll_order,
            layer,
        ):
            print(f"Interpolating layer: {layer}")
            coefficients = np.zeros(
                shape=(new_coordinates[0].shape[0], original_mesh.n_gll_points)
            )
            element = np.empty(new_coordinates[0].shape[0], dtype=np.int)
            nodes = original_mesh.points[original_mask[layer]]
            for _i, coord in enumerate(new_coordinates[0]):
                element[_i], ref_coord = _check_if_inside_element(
                    gll_model=nodes,
                    nearest_elements=nearest_elements[_i, :],
                    point=coord,
                    dimension=dimensions,
                    ignore_hard_elements=True,
                )
                coefficients[_i] = get_coefficients(
                    a=from_gll_order,
                    b=from_gll_order,
                    c=from_gll_order,
                    ref_coord=ref_coord,
                    dimension=dimensions,
                )
            return coefficients, element

        coeffs[layer], elements[layer] = _fill_value_array(
            new_coordinates=unique_new_points[layer],
            nearest_elements=nearest_element_indices,
            original_mesh=original_mesh,
            original_mask=original_mask,
            parameters=parameters,
            dimensions=dimensions,
            from_gll_order=from_gll_order,
            layer=layer,
        )

    if loop:
        if threads is None:
            threads = multiprocessing.cpu_count()
        print(f"Solving problem using {threads} threads")
        layer_list = list(unique_new_points.keys())
        threads = min(threads, len(layer_list))

        with multiprocessing.Pool(threads) as pool:
            pool.map(_find_interpolation_weights, layer_list)
        pool.close()
        pool.join()

        if stored_array is not None:
            print("Saving interpolation matrices")
            dataset = h5py.File(
                os.path.join(stored_array, "interp_info.h5"), "w"
            )
            for k, v in coeffs.items():
                dataset.create_dataset(f"coeffs/{k}", data=v)
            for k, v in elements.items():
                dataset.create_dataset(f"elements/{k}", data=v)
            dataset.close()
    else:
        print("No need to loop, we have weights")

    for layer in coeffs.keys():
        # num_failed += len(np.where(elements[layer] == -1)[0])
        elms = elements[layer].astype(int)

        for param in parameters:
            values = np.sum(
                original_mesh.element_nodal_fields[param][
                    original_mask[layer]
                ][elms]
                * coeffs[layer],
                axis=1,
            )
            new_mesh.element_nodal_fields[param][mask[layer]] = values[
                unique_new_points[layer][1]
            ].reshape(new_mesh.element_nodal_fields[param][mask[layer]].shape)
    for _i, param in enumerate(parameters):
        new_field = np.zeros_like(new_mesh.element_nodal_fields[param])
        for layer in coeffs.keys():
            elms = elements[layer].astype(int)
            values = np.sum(
                original_mesh.element_nodal_fields[param][
                    original_mask[layer]
                ][elms]
                * coeffs[layer],
                axis=1,
            )
            new_field[mask[layer]] = values[
                unique_new_points[layer][1]
            ].reshape(new_mesh.element_nodal_fields[param][mask[layer]].shape)
        new_mesh.attach_field(name=param, data=new_field)


def gll_2_gll(
    from_gll,
    to_gll,
    nelem_to_search=20,
    parameters="ISO",
    from_model_path="MODEL/data",
    to_model_path="MODEL/data",
    from_coordinates_path="MODEL/coordinates",
    to_coordinates_path="MODEL/coordinates",
    gradient=False,
    stored_array=None,
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
    :param to_gll: path to gll mesh to interpolate to
    :param dimensions: dimension of meshes.
    :param nelem_to_search: amount of elements to check
    :param parameters: Parameters to be interpolated, possible to pass, "ISO",
    "TTI" or a list of parameters.
    :param gradient: If this is a gradient to be added to another gradient,
    only put true if you want to add on top of a currently existing gradient
    :param stored_array: If you want to store the array for future
    interpolations. If the array exists in that path it will be loaded. Store
    elements under elements.npy and coeffs under coeffs.npy
    """

    print("Initialization stage")
    print(f"Stored array: {stored_array}")
    (
        original_points,
        original_data,
        original_params,
    ) = utils.load_hdf5_params_to_memory(
        from_gll, from_model_path, from_coordinates_path
    )

    dimensions = original_points.shape[2]
    from_gll_order = (
        int(round(original_data.shape[2] ** (1.0 / dimensions))) - 1
    )
    parameters = original_params
    # parameters = utils.pick_parameters(parameters)
    assert set(parameters) <= set(
        original_params
    ), f"Original mesh does not have all the parameters you wish to interpolate. You asked for {parameters}, mesh has {original_params}"

    all_old_points = original_points.reshape(
        original_points.shape[0] * original_points.shape[1],
        original_points.shape[2],
    )
    original_tree = KDTree(all_old_points)
    new = h5py.File(to_gll, "r+")

    # We look for the fluid elements, we wan't to avoid solids getting fluid values
    # which can happen if one gll point hits a solid value.
    new_points = np.array(new[to_coordinates_path][:], dtype=np.float64)
    elem_params = (
        new["MODEL/element_data"].attrs.get("DIMENSION_LABELS")[1].decode()
    )
    elem_params = elem_params[2:-2].replace(" ", "").split("|")
    fluid_index = elem_params.index("fluid")
    fluid_elements = new["MODEL/element_data"][:, fluid_index].astype(bool)
    solid_elements = np.invert(fluid_elements)
    # Save the current values in order to fix any case of solid getting fluid values
    new_values = np.copy(new[to_model_path][:])

    permutation = np.arange(0, len(parameters))
    i = 0
    for param in original_params:
        if param in parameters:
            permutation[i] = parameters.index(param)
            i += 1
    """
    The reordering of parameters is currently not used, what is used is simply to
    interpolate all parameters from the parent mesh to the receiving mesh
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
        print("I have to reorder parameters")
        args = np.argsort(permutation).astype(int)
    else:
        args = np.arange(start=0, stop=len(permutation)).astype(int)
    parameters = [parameters[x] for x in args]
    print(parameters)
    """

    gll_points = new[to_coordinates_path].shape[1]
    loop = True
    if stored_array:
        if os.path.exists(os.path.join(stored_array, "coeffs.npy")):
            coeffs = np.load(
                os.path.join(stored_array, "coeffs.npy"), allow_pickle=True
            )
            if os.path.exists(os.path.join(stored_array, "elements.npy")):
                element = np.load(
                    os.path.join(stored_array, "elements.npy"),
                    allow_pickle=True,
                )
                loop = False
                k = np.isnan(coeffs)
                # print(f"NAN DETECTED for coeffs: {np.where(k)}")
                # print(f"AMOUNT OF NANS: {np.where(k)[0].shape}")
                assert np.where(k)[0].shape[0] == 0, print(
                    "Stored coeffs matrix has NaNs"
                )
    # Prepare all the points in order to loop through it faster.
    # Points are prepared in a way thet we find unique gll points
    # and loop through those to save time.
    unique_new_points, recon = utils.get_unique_points(points=new_points)

    if loop:
        nearest_element_indices = np.zeros(
            shape=[unique_new_points.shape[0], nelem_to_search], dtype=np.int
        )

        _, nearest_element_indices[:, :] = original_tree.query(
            unique_new_points[:, :], k=nelem_to_search
        )
        nearest_element_indices = np.floor(
            nearest_element_indices / original_points.shape[1]
        ).astype(int)

        nearest_element_indices = np.swapaxes(nearest_element_indices, 0, 1)
        unique_new_points = np.swapaxes(unique_new_points, 0, 1)
        coeffs_empty = np.zeros(
            shape=[
                len(parameters),
                original_points.shape[1],
                unique_new_points.shape[1],
            ]
        )

        element_empty = np.zeros(shape=unique_new_points.shape[1])

        print("Now we start interpolating")
        # from multiprocessing import Process

        element, coeffs = find_gll_coeffs(
            original_coordinates=original_points,
            coordinates=unique_new_points,
            nearest_elements=nearest_element_indices,
            coeffs=coeffs_empty,
            element=element_empty,
            dimensions=dimensions,
            from_gll_order=from_gll_order,
            ignore_hard_elements=True,
        )

        # k = np.isnan(element)
        # print(f"NAN DETECTED for elements: {np.where(k)}")
        k = np.isnan(coeffs)
        print(f"NAN DETECTED for coeffs: {np.where(k)}")
        print(f"AMOUNT OF NANS: {np.where(k)[0].shape}")
        assert np.where(k)[0].shape[0] == 0, print(
            "Interpolation failed somehow"
        )
        for i in range(len(parameters))[1:]:
            coeffs[i, :, :] = coeffs[0, :, :]
        print(
            "Interpolation done, Need to organize the results and write to file"
        )
        num_failed = len(np.where(element == -1)[0])
        if num_failed > 0:
            print(f"{num_failed} points could not find an enclosing element.")
            element = element.astype(int)
        if stored_array:
            if not os.path.exists(stored_array):
                os.makedirs(stored_array)
            print("Will save matrices for later usage")
            np.save(
                os.path.join(stored_array, "elements.npy"),
                element,
                allow_pickle=True,
            )
            np.save(
                os.path.join(stored_array, "coeffs.npy"),
                coeffs,
                allow_pickle=True,
            )
    else:
        print("Matrix was already stored. Will use that one")

    resample_data = original_data[element]
    # k = np.isnan(resample_data)
    # print(f"NAN DETECTED for resample_data: {np.where(k)}")
    # Reorder everything to be able to save to file in correct format.
    coeffs = np.swapaxes(coeffs, 0, 2).swapaxes(1, 2)
    print(resample_data.shape)
    print(f"Resample_data: {resample_data.shape}")
    print(f"Coeffs: {coeffs.shape}")
    values = (
        np.sum(resample_data[:, :, :] * coeffs[:, :, :], axis=2)[recon, :]
        .reshape((new_points.shape[0], gll_points, len(parameters)))
        .swapaxes(1, 2)
    )
    k = np.isnan(values)
    print(f"NAN DETECTED for values: {np.where(k)}")
    if not gradient:
        values[~solid_elements] = new_values[~solid_elements]

        if "VS" in parameters:
            vs_index = parameters.index("VS")
        else:
            vs_index = parameters.index("VSV")
        # look at fake fluid values
        zero_vs = np.where(values[:, vs_index, :] == 0.0)
        print(
            "If any fluid values accidentally went to the solid part we fix it"
        )
        for _i, elem in enumerate(np.unique(zero_vs[0])):
            if solid_elements[elem]:
                values[elem, :, :] = new_values[elem, :, :]

    # This needs to be implemented as a sum not gradient.
    # if gradient:
    #     # Gradient implementations still need to be looked at.
    #     existing = new[to_model_path]
    #     values += existing
    utils.remove_and_create_empty_dataset(
        new, parameters, to_model_path, to_coordinates_path
    )

    new[to_model_path][:, :, :] = values


def interpolate_to_points_layered(
    from_mesh,
    to_mesh,
    parameters,
    layers="nocore",
    make_spherical=False,
    nelem_to_search=20,
):
    """
    A more stable version of interpolate_to_points, given that the two meshes
    have the same layers. That the meshes were made using the same 1D mesh.

    :param mesh: [description]
    :type mesh: [type]
    :param points: [description]
    :type points: [type]
    :param params_to_interp: [description]
    :type params_to_interp: [type]
    :param make_spherical: [description], defaults to False
    :type make_spherical: bool, optional
    """
    from multi_mesh.components.salvus_mesh_reader import SalvusMesh

    original_mesh = SalvusMesh(from_mesh, fast_mode=False)
    original_mask, layers = utils.create_layer_mask(
        mesh=original_mesh, layers=layers
    )
    if parameters == "all":
        parameters = list(original_mesh.element_nodal_fields.keys())
    dimensions = 3

    new_mesh = SalvusMesh(to_mesh, fast_mode=False)

    # Unique new points is a dictionary with tuples of coordinates and a
    # reconstruction array
    unique_new_points, mask, layers = utils.get_unique_points(
        points=new_mesh, layers=layers
    )
    parameters = utils.pick_parameters(parameters)
    nearest_element_indices = {}
    original_trees = {}
    gll_order = original_mesh.shape_order
    values = np.zeros(shape=())
    for layer in layers:
        layer = str(layer)
        points = original_mesh.get_element_centroids()[original_mask[layer]]
        original_trees[layer] = KDTree(points)
        _, nearest_element_indices[layer] = original_trees[layer].query(
            unique_new_points[layer][0], k=nelem_to_search
        )
    elem_indices, coeffs = get_element_weights_layered(
        new_coordinates=unique_new_points,
        original_mesh=original_mesh,
        original_mask=original_mask,
        nearest_elements=nearest_element_indices,
        from_gll_order=gll_order,
        dimensions=dimensions,
    )
    num_failed = 0
    for _i, param in enumerate(parameters):
        new_field = np.zeros_like(new_mesh.element_nodal_fields[param])
        for layer in coeffs.keys():
            elms = elem_indices[layer].astype(int)
            if _i == 0:
                num_failed += len(np.where(elem_indices[layer] == -1)[0])
            values = np.sum(
                original_mesh.element_nodal_fields[param][
                    original_mask[layer]
                ][elms]
                * coeffs[layer],
                axis=1,
            )
            new_field[mask[layer]] = values[
                unique_new_points[layer][1]
            ].reshape(new_mesh.element_nodal_fields[param][mask[layer]].shape)
        new_mesh.attach_field(name=param, data=new_field)
    if num_failed > 0:
        print(f"{num_failed} points could not be interpolated")


def interpolate_to_points(
    mesh, points, params_to_interp, make_spherical=False
):
    """
    Interpolates from a mesh to point cloud.

    :param mesh: Mesh from which you want to interpolate
    :param points: np.array of points that require interpolation,
    if they are not found. zero is returned
    :param params_to_interp: list of params to interp
    :param make_spherical: bool that determines if mesh gets mapped to a sphere.
    :return: array[nparams_to_interp, npoints]
    """

    if make_spherical:
        map_to_sphere(mesh)
    if isinstance(mesh, str):
        from salvus.mesh.unstructured_mesh import UnstructuredMesh

        mesh = UnstructuredMesh.from_h5(mesh)
    elem_centroid = mesh.get_element_centroid()
    print("Initializing KDtree...")
    centroid_tree = KDTree(elem_centroid)

    # Get GLL points from old mesh
    gll_points = mesh.points[mesh.connectivity]
    gll_order = mesh.shape_order

    # Get elements and interpolation coefficients for new_points
    print("Retrieving interpolation weights")
    elem_indices, coeffs = get_element_weights(
        gll_points, gll_order, centroid_tree, points
    )

    num_failed = len(np.where(elem_indices == -1)[0])
    if num_failed > 0:
        print(
            num_failed,
            "points could not find an enclosing element. "
            "These points will be set to zero. "
            "Please check your domain or the interpolation tuning parameters",
        )

    print("Interpolating fields...")
    vals = np.zeros((len(points), len(params_to_interp)))
    for i, param in enumerate(params_to_interp):
        old_element_nodal_vals = mesh.element_nodal_fields[param]
        vals[:, i] = np.sum(
            coeffs * old_element_nodal_vals[elem_indices], axis=1
        )
    return vals


def map_to_ellipse(base_mesh, mesh):
    """Takes a base mesh with ellipticity topography and
    stretches the mesh to have the same ellipticity.

    # TODO, this could also be merged with interpolate functions, such
    # TODO that weights do not need to be computed twice
    """
    # Get radial ratio for each element node
    r_earth = 6371000
    r = np.sqrt(np.sum(base_mesh.points ** 2, axis=1)) / r_earth
    _, i = np.unique(base_mesh.connectivity, return_index=True)
    rad_1d_values = base_mesh.element_nodal_fields["z_node_1D"].flatten()[i]
    r_ratio = r / rad_1d_values
    r_ratio_element_nodal_base = r_ratio[base_mesh.connectivity]

    # Map to sphere and store original points
    orig_old_elliptic_mesh_points = np.copy(base_mesh.points)
    map_to_sphere(base_mesh)
    map_to_sphere(mesh)

    # For each point in new mesh find nearest elements centroids in old mesh
    elem_centroid = base_mesh.get_element_centroid()
    centroid_tree = KDTree(elem_centroid)
    gll_points = base_mesh.points[base_mesh.connectivity]

    # Get elements and interpolation coefficients for new_points
    print("Retrieving interpolation weigts")
    elem_indices, coeffs = get_element_weights(
        gll_points, centroid_tree, mesh.points
    )

    num_failed = len(np.where(elem_indices == -1)[0])
    if num_failed > 0:
        raise Exception(
            f"{num_failed} points could not find an enclosing element."
        )

    mesh_point_r_ratio = np.sum(
        coeffs * r_ratio_element_nodal_base[elem_indices], axis=1
    )
    mesh.points = np.array(mesh_point_r_ratio * mesh.points.T).T
    base_mesh.points = orig_old_elliptic_mesh_points


def map_to_sphere(mesh):
    """
    Takes a salvus mesh and converts it to a sphere.
    Acts on the passed object
    """
    if isinstance(mesh, salvus.mesh.unstructured_mesh.UnstructuredMesh):
        _, i = np.unique(mesh.connectivity, return_index=True)
        rad_1D = mesh.element_nodal_fields["z_node_1D"].flatten()[i]
    else:
        rad_1D = mesh.element_nodal_fields["z_node_1D"]

    r_earth = 6371000
    x, y, z = mesh.points.T
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Conert all points that do not lie right in the core
    # I think this should work for the SalvusMesh too
    x[r > 0] = x[r > 0] * r_earth * rad_1D[r > 0] / r[r > 0]
    y[r > 0] = y[r > 0] * r_earth * rad_1D[r > 0] / r[r > 0]
    z[r > 0] = z[r > 0] * r_earth * rad_1D[r > 0] / r[r > 0]


def get_element_weights(gll_points, shape_order, centroid_tree, points):
    """
    A function to figure out inside which element the point to be
    interpolated is. In addition, it gives the interpolation coefficients
    for the respective element.
    Returns  -1 and no coeffs when nothing is found.

    :param gll: All GLL nodes from old_mesh
    :param centroid_tree: scipy.spatial.cKDTree that is initialized with the
     centroids of the elements of old_mesh
    :param points: List of points that require interpolation
    :return: the enclosing elements and interpolation weights
    """
    global _get_coeffs
    nelem_to_search = 25

    def _get_coeffs(point_indices):
        _, nearest_elements = centroid_tree.query(
            points[point_indices], k=nelem_to_search
        )
        element_num = np.arange(len(point_indices))

        def check_inside(index, element_num):
            """
            returns the element_id and coefficients for new_points[index]
            returns -1 for index when nothing is found
            """

            for element in nearest_elements[element_num]:
                # get element gll_points
                gll_points_elem = np.asfortranarray(
                    gll_points[element, :, :], dtype=np.float64
                )
                point = np.asfortranarray(points[index])

                ref_coord = inverse_transform(
                    point, gll_points=gll_points_elem, dimension=3
                )

                # tolerance of 3%
                if np.any(np.isnan(ref_coord)):
                    continue

                if np.all(np.abs(ref_coord) < 1.03):
                    coeffs = get_coefficients(
                        shape_order,
                        0,
                        0,
                        np.asfortranarray(ref_coord, dtype=np.float64),
                        3,
                    )
                    return element, coeffs
            # return weights zero if nothing found
            return -1, np.zeros((shape_order + 1) ** 3)

        a = np.vectorize(
            check_inside, signature="(),()->(),(n)", otypes=[int, float]
        )
        return a(point_indices, element_num)

    # Split array in chunks
    num_processes = multiprocessing.cpu_count()
    n = 50 * num_processes
    task_list = np.array_split(np.arange(len(points)), n)

    elems = []
    coeffs = []
    with multiprocessing.Pool(num_processes) as pool:
        with tqdm(
            total=len(task_list),
            bar_format="{l_bar}{bar}[{elapsed}<{remaining},"
            " '{rate_fmt}{postfix}]",
        ) as pbar:
            for i, r in enumerate(pool.imap(_get_coeffs, task_list)):
                elem_in, coeff = r
                pbar.update()
                elems.append(elem_in)
                coeffs.append(coeff)
        pool.close()
        pool.join()

    elems = np.concatenate(elems)
    coeffs = np.concatenate(coeffs)
    return elems, coeffs


def get_element_weights_layered(
    new_coordinates: Dict[str, np.ndarray],
    nearest_elements: Dict[str, np.ndarray],
    original_mesh: salvus.mesh.unstructured_mesh.UnstructuredMesh,
    original_mask: Dict[str, np.ndarray],
    dimensions: int = 3,
    from_gll_order: int = 2,
):
    global _get_coeffs_layered

    def _get_coeffs_layered(point_indices):
        # element_num = np.arange(len(point_indices))
        # TODO: This might only work with a fresh centroid tree query!
        def check_inside(new_point_index):
            point = np.asfortranarray(
                new_coordinates[layer][0][new_point_index], dtype=np.float64
            )
            for element in nearest_elements[layer][new_point_index]:
                gll_points_old = np.asfortranarray(
                    original_mesh.points[original_mask[layer]][element, :, :],
                    dtype=np.float64,
                )
                ref_coord = inverse_transform(
                    point=point,
                    gll_points=gll_points_old,
                    dimension=dimensions,
                )

                if np.any(np.isnan(ref_coord)):
                    continue
                if np.all(np.abs(ref_coord) < 1.03):
                    coeffs = get_coefficients(
                        from_gll_order,
                        0,
                        0,
                        np.asfortranarray(ref_coord, dtype=np.float64),
                        3,
                    )
                    return element, coeffs
            return -1, np.zeros((from_gll_order + 1) ** dimensions)

        a = np.vectorize(
            check_inside, signature="()->(),(n)", otypes=[int, float]
        )
        return a(point_indices)

    # Now I need the multiprocessing magic
    # First I'll try to do this as a loop through the layers
    num_cpus = multiprocessing.cpu_count()
    print(f"Num cpus: {num_cpus}")
    elems = {}
    coeffs = {}
    for layer, point in new_coordinates.items():
        element_list = []
        coeff_list = []
        factor = np.round(len(point[0]) / num_cpus / 4.0)
        print(f"Factor: {factor}")
        points = np.array_split(np.arange(len(point[0])), num_cpus * factor)
        chunksize = len(points[0])
        print(f"Chunksize: {chunksize}")
        with multiprocessing.Pool(num_cpus) as pool:
            print(f"Layer: {layer}")
            with tqdm(
                total=len(points),
                bar_format="{l_bar}{bar}[{elapsed}<{remaining},"
                " '{rate_fmt}{postfix}]",
            ) as pbar:
                for r in pool.imap(
                    _get_coeffs_layered, points, chunksize=chunksize
                ):
                    elem_in, coeff = r
                    pbar.update()
                    element_list.append(elem_in)
                    coeff_list.append(coeff)
            pool.close()
            pool.join()

        elems[layer] = np.concatenate(element_list)
        coeffs[layer] = np.concatenate(coeff_list)
        print(f"Done with layer: {layer}")
    return elems, coeffs


def get_coefficients(a, b, c, ref_coord, dimension):

    if dimension == 3:
        if a == 4:
            return GetInterpolationCoefficients3D_order_4(ref_coord)
        elif a == 2:
            return GetInterpolationCoefficients3D_order_2(ref_coord)
        elif a == 1:
            return GetInterpolationCoefficients3D_order_1(ref_coord)
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

    if dimension == 3:
        if len(gll_points) == 125:
            return salvus.fem.hypercube.InverseCoordinateTransformWrapper(
                n=4, d=3, pnt=point, ctrlNodes=gll_points
            )
        if len(gll_points) == 27:
            return InverseCoordinateTransformWrapper3D_2(
                pnt=point, ctrlNodes=gll_points
            )
        if len(gll_points) == 8:
            return InverseCoordinateTransformWrapper3D_1(
                pnt=point, ctrlNodes=gll_points
            )
    elif dimension == 2:
        return InverseCoordinateTransformWrapper2D(
            pnt=point, ctrlNodes=gll_points
        )


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
        centroids[:, d] = np.mean(
            gll_coordinates[:, :, d], axis=1, dtype=np.float64
        )

    return centroids


def _check_if_inside_element(
    gll_model, nearest_elements, point, dimension, ignore_hard_elements
):
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
    inside = np.zeros(len(nearest_elements), dtype=bool)
    # print(f"Nearest elements: {nearest_elements}")
    # print(f"Nearest shape: {nearest_elements.shape}")
    # sys.exit("stop now")
    for _i, element in enumerate(nearest_elements):
        gll_points = gll_model[element, :, :]
        gll_points = np.asfortranarray(gll_points)
        inside[_i], dist[_i] = boundary_box_check(point, gll_points)
        if inside[_i]:
            ref_coord = inverse_transform(
                point=point, gll_points=gll_points, dimension=dimension
            )
            if np.any(np.isnan(ref_coord)):
                continue

            if np.all(np.abs(ref_coord) <= 1.02):
                return element, ref_coord
    # if not ignore_hard_elements:
    #     warnings.warn(
    #         "Could not find an element which this points fits into."
    #         " Maybe you should add some tolerance."
    #         " Will return the best searched element"
    #     )

    if np.any(inside):
        ind = np.where(inside)
        ind = np.where(dist == np.min(dist[ind]))[0][0]
    else:
        ind = np.where(dist == np.min(dist))[0][0]
    if ind is None:
        ind = 0
    # if ind >= 20:
    # ind = 0
    # # ind = ref_coords.index(ref_coords == np.min(ref_coords))
    element = nearest_elements[ind]

    ref_coord = inverse_transform(
        point=point,
        gll_points=np.asfortranarray(
            gll_model[element, :, :], dtype=np.float64
        ),
        dimension=dimension,
    )
    if np.any(np.isnan(ref_coord)):
        if not ignore_hard_elements:
            raise ValueError("Can't find an appropriate element.")
        ref_coord = np.array([0.645, -0.5, 0.22])
    if np.any(np.abs(ref_coord) >= 1.02):
        # Assign a random coordinate in best fitting element
        ref_coord = np.array([0.645, -0.5, 0.22])
        return -1, np.zeros(3)
    return element, ref_coord


def fill_value_array(
    new_coordinates: Dict[str, np.ndarray],
    nearest_elements: Dict[str, np.ndarray],
    original_mesh: salvus.mesh.unstructured_mesh.UnstructuredMesh,
    original_mask: Dict[str, np.ndarray],
    parameters: List[str],
    dimensions: int = 3,
    from_gll_order: int = 2,
):
    """
    Similar to find_gll_coeffs except this one fills the parameter matrix
    on the fly, thus getting rid of the memory intensive matrix computations
    at the end of the loop.

    :param original_coordinates: An array of coordinates from the original
        mesh
    :type original_coordinates: numpy.ndarray
    :param new_coordinates: An array of coordinates from the new mesh
    :type new_coordinates: numpy.ndarray
    :param nearest_elements: An array describing which are the closest
        elements to the point we want to find the value for
    :type nearest_elements: numpy.ndarray
    :param original_mesh: salvus mesh object for original mesh
    :type original_mesh: salvus.mesh.unstructured_mesh.UnstructuredMesh
    :param original_mask: boolean array used to mask the points of the
        original mesh
    :type original_mask: numpy.ndarray
    :param new_values: An empty array of values to fill, we need to relate
        this to parameters in some way
    :type new_values: numpy.ndarray
    :param dimensions: How many dimensions there are in the mesh, defaults
        to 3
    :type dimensions: int, optional
    :param from_gll_order: The gll order of the original mesh, defaults to 2
    :type from_gll_order: int, optional
    """

    element = {}
    coeffs = {}

    # nodes = original_mesh.get_element_nodes()[original_mask]
    for key, val in new_coordinates.items():
        print(f"Interpolating layer: {key}")
        coeffs[key] = np.zeros(
            shape=(val[0].shape[0], original_mesh.n_gll_points)
        )
        element[key] = np.empty(val[0].shape[0], dtype=int)
        nodes = original_mesh.points[original_mask[key]]
        for _i, coord in tqdm(enumerate(val[0]), total=val[0].shape[0]):
            element[key][_i], ref_coord = _check_if_inside_element(
                gll_model=nodes,
                nearest_elements=nearest_elements[key][_i],
                point=coord,
                dimension=dimensions,
                ignore_hard_elements=True,
            )
            coeffs[key][_i] = get_coefficients(
                a=from_gll_order,
                b=from_gll_order,
                c=from_gll_order,
                ref_coord=ref_coord,
                dimension=dimensions,
            )
    return coeffs, element


def find_gll_coeffs(
    original_coordinates: np.array,
    coordinates: np.array,
    nearest_elements: np.array,
    coeffs: np.array,
    element: np.array,
    dimensions: int,
    from_gll_order: int,
    ignore_hard_elements: bool,
):
    """
    Loop through coordinates, figure out which elements they are in and
    compute the relevent gll_coefficients. Returns element indices and
    corresponding coefficients.

    :param original_coordinates: An array of coordinates from the hdf5 file
    which you want to interpolate from
    :type original_coordinates: np.array
    :param coordinates: An array of coordinates from the hdf5 file which
    you want to interpolate to
    :type coordinates: np.array
    :param nearest_elements: An array of closest elements from KD tree
    :type nearest_elements: np.array
    :param dimensions: Spatial dimension of the problem
    :type dimensions: int
    :param from_gll_order: Polynomial order of mesh to interpolate from
    :type from_gll_order: int
    :param ignore_hard_elements: Sometimes there are problems with funky
    smoothiesem elements, this ignores them and gives its best value
    :type ignore_hard_elements: bool
    """
    for i in tqdm(range(coordinates.shape[1])):
        element[i], ref_coord = _check_if_inside_element(
            original_coordinates,
            nearest_elements[:, i],
            coordinates[:, i],
            dimensions,
            ignore_hard_elements,
        )
        if np.max(np.abs(ref_coord)) > 1.3:
            print(f"REF_COORD IS NAN!!: {ref_coord}")
        if element[i] == -1:
            -1, np.zeros(3)
            coeffs[0, :, i] = np.zeros((from_gll_order + 1) ** dimensions)
        else:
            coeffs[0, :, i] = get_coefficients(
                from_gll_order,
                from_gll_order,
                from_gll_order,
                ref_coord,
                dimensions,
            )
        # if np.any(coeffs[0, :, i] >= 2.0):
        #     print(f"coeffs are big! {np.max(np.abs(coeffs))} \n")
        #     print(ref_coord)
        #     print(ref_coord.type)

    return element, coeffs


def extract_regular_grid(
    mesh: Union[
        str, pathlib.Path, salvus.mesh.unstructured_mesh.UnstructuredMesh
    ],
    parameters: List[str],
    lat_extent: Tuple[float, float, float],
    lon_extent: Tuple[float, float, float],
    rad_extent: Tuple[float, float, float],
):
    """
    Salvus meshes live on unregular grids, this is a way to extract a regular
    grid based on a salvus mesh

    :param mesh: The mesh object or a path to it
    :type mesh: Union[str, 
        pathlib.Path, salvus.mesh.unstructured_mesh.UnstructuredMesh]
    :param parameters: Name of parameters to interpolate
    :type parameters: List[str]
    :param lat_extent: min_latitude, max_latitude, num_points
    :type lat_extent: Tuple[float, float, float]
    :param lon_extent: min_longitude, max_longitude, num_points
    :type lon_extent: Tuple[float, float, float]
    :param rad_extent: min_radius, max_radius in meters, num_points
    :type rad_extent: Tuple[float, float, float]
    """
    from salvus.mesh.unstructured_mesh_utils import (
        extract_model_to_regular_grid,
    )

    if isinstance(mesh, (str, pathlib.Path)):
        from salvus.mesh.unstructured_mesh import UnstructuredMesh as um

        mesh = um.from_h5(mesh)

    lat = np.linspace(
        start=lat_extent[0], stop=lat_extent[1], num=lat_extent[2]
    )
    lon = np.linspace(
        start=lon_extent[0], stop=lon_extent[1], num=lon_extent[2]
    )
    radius = np.linspace(
        start=rad_extent[0], stop=rad_extent[1], num=rad_extent[2]
    )
    ds = utils.create_xarray_dataset(lat=lat, lon=lon, radius=radius)

    ds = extract_model_to_regular_grid(
        mesh=mesh, ds=ds, pars=parameters, verbose=True,
    )

    return ds
