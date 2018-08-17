import click

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@click.group()
def cli():
    pass

@cli.command()
@click.option('--mesh_a', help="Salvus continuous exodus file.", required=True)
@click.option('--mesh_b', help="Salvus continuous exodus file.", required=True)
@click.option('--param', help="parameter to interpolate.", required=True)
def interpolate_mesh_a_to_b(mesh_a, mesh_b, param):
    """Interpolates values from mesh A onto mesh B, exodus to exodus"""
    from multi_mesh.io.exodus import Exodus
    from scipy.spatial import cKDTree
    import numpy as np
    from multi_mesh.helpers import load_lib
    lib = load_lib()

    # Read Mesh A (exodus format)
    exodus_a = Exodus(mesh_a)

    # Create KDTree from mesh a elemenet centroids
    a_centroids = exodus_a.get_element_centroid()
    centroid_tree = cKDTree(a_centroids, balanced_tree=False)

    # Read Mesh B and search for nearest_element_indices
    nelem_to_search = 20
    exodus_b = Exodus(mesh_b, mode="a")
    _, nearest_element_indices = centroid_tree.query(exodus_b.points, k=nelem_to_search)

    npoints = exodus_b.npoint  # npoints that require interpolation
    enclosing_elem_node_indices = np.zeros((npoints, 8), dtype=np.int64)  # indices of the enclosing element
    weights = np.zeros((npoints, 8))  # interpolation weights
    permutation = [0, 3, 2, 1, 4, 5, 6, 7]
    i = np.argsort(permutation)
    connectivity_reordered = exodus_a.connectivity[:, i]

    nfailed = lib.triLinearInterpolator(nelem_to_search,
                                        npoints,
                                        nearest_element_indices,
                                        np.ascontiguousarray(connectivity_reordered),
                                        enclosing_elem_node_indices,
                                        np.ascontiguousarray(exodus_a.points),
                                        weights,
                                        np.ascontiguousarray(exodus_b.points))

    param_a = exodus_a.get_nodal_field(param)
    values = np.sum(param_a[enclosing_elem_node_indices] * weights, axis=1)
    exodus_b.attach_field(param, np.zeros_like(values))
    exodus_b.attach_field(param, values)

    assert nfailed is 0, f"{nfailed} points could not be interpolated."


@cli.command()
@click.option('--mesh', help="Salvus continuous exodus file.", required=True)
@click.option('--gll_model', help="Salvus continuous exodus file.", required=True)
@click.option('--gll_order', help="Order of polynomials inside your gll "
                                  "model", default=4)
@click.option('--param', help="parameter to interpolate.", required=False)
def interpolate_mesh_to_gll(mesh, gll_model, gll_order, param):
    """
    Interpolate values from normal exodus mesh to a smoothiesem gll model
    Keep this isotropic for now
    """
    from multi_mesh.io.exodus import Exodus
    from scipy.spatial import cKDTree
    import numpy as np
    from multi_mesh.helpers import load_lib
    import h5py
    import time
    lib = load_lib()
    start = time.time()
    # Read in exodus mesh
    exodus = Exodus(mesh)
    a_centroids = exodus.get_element_centroid()
    centroid_tree = cKDTree(a_centroids, balanced_tree=False)

    # Read in gll model
    gll = h5py.File(gll_model, 'r+')

    # find coordinates in gll_model
    gll_coords = gll['MODEL']['coordinates']
    gll_points = (gll_order + 1) ** 3

    nelem_to_search = 20
    nearest_element_indices = np.zeros(shape=[gll_points, len(gll_coords), nelem_to_search], dtype=np.int64)

    for i in range(gll_points):
        print(f"Finding element indices for gll point: {i+1}/{gll_points}")
        _, nearest_element_indices[i, :, :] = centroid_tree.query(gll_coords[:, i, :], k=nelem_to_search)

    npoints = len(gll_coords)

    enclosing_elem_node_indices = np.zeros((gll_points, npoints, 8), dtype=np.int64)
    weights = np.zeros((gll_points, npoints, 8))
    permutation = [0, 3, 2, 1, 4, 5, 6, 7]
    i = np.argsort(permutation)
    connectivity_reordered = exodus.connectivity[:, i]

    for i in range(gll_points):
        print(f"Trilinear interpolation for gll point: {i+1}/{gll_points}")
        nfailed = lib.triLinearInterpolator(nelem_to_search,
                                            npoints,
                                            nearest_element_indices[i, :, :],
                                            np.ascontiguousarray(connectivity_reordered),
                                            enclosing_elem_node_indices[i, :, :],
                                            np.ascontiguousarray(exodus.points),
                                            weights[i, :, :],
                                            np.ascontiguousarray(gll_coords[:, i, :]))

    # Lets just interpolate the first parameter

    param_a = 'VPV'
    param = exodus.get_nodal_field(param_a)
    for i in range(gll_points):
        print(f"Putting values onto gll points: {i+1}/{gll_points}")
        values = np.sum(param[enclosing_elem_node_indices[i, :, :]] * weights[i, :, :], axis=1)


        gll['MODEL']['data'][:, 2, i] = values
    end = time.time()
    runtime = end - start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime} minutes")
    else:
        print(f"Finished in time: {runtime} seconds")

    assert nfailed is 0, f"{nfailed} points could not be interpolated."



@cli.command()
@click.option('--mesh', help="Salvus continuous exodus file.", required=True)
@click.option('--gll_model', help="Salvus continuous exodus file.", required=True)
# @click.option('--gll_mesh', help="An exodus file containing.", required=True)
# @click.option('--gll_order', help="Order of polynomials inside your gll "
#                                   "model", default=4)
# @click.option('--param', help="parameter to interpolate.", required=False)
def interpolate_gll_to_mesh(mesh, gll_model):
    """
    A function which takes model parameters stored on GLL model and
    interpolates them on to a nodal mesh
    :param mesh: name of meshfile
    :param gll_model: name of gll_model file
    :param gll_mesh: a mesh file with the elements of the gll model
    :param gll_order: order of lagrange polynomials
    :param param: which parameter to interpolate, will default to all.
    :return:
    """

    from multi_mesh.io.exodus import Exodus
    from scipy.spatial import cKDTree
    import numpy as np
    import h5py
    import time
    import salvus_fem
    start = time.time()

    # Read in gll model
    # gll = h5py.File(gll_model, 'r')
    # Compute centroids
    centroids = _find_gll_centroids(gll_model, 3)
    # Build a KDTree of the centroids to look for nearest elements
    print("Building KDTree")
    centroid_tree = cKDTree(centroids, balanced_tree=True)
    print("KDTree is built")

    nelem_to_search = 30
    # Read in mesh
    print("Read in mesh")
    exodus = Exodus(mesh, mode="a")
    # Find nearest elements
    print("Querying the KDTree")
    _, nearest_element_indices = centroid_tree.query(exodus.points, k=nelem_to_search)
    npoints = exodus.npoint
    values = np.zeros(shape=[npoints])

    s = 0
    for point in exodus.points:
        print(s)
        print(f"Nearest element indices: {nearest_element_indices[s,:]}")
        element, ref_coord = _check_if_inside_element(gll_model,
                                           nearest_element_indices[s, :],
                                           point)

        coeffs = salvus_fem.getInterpolationCoefficients(4, 4, 4, "MATRIX", "MATRIX", ref_coord)
        with h5py.File(gll_model, 'r') as gll:
            values[s] = gll['MODEL']['DATA'][element, 0, :] * coeffs

        s += 1

    param = "New_stuff"
    exodus.attach_field(param, np.zeros_like(values))
    exodus.attach_field(param, values)

    end = time.time()
    runtime = end-start

    if runtime >= 60:
        runtime = runtime / 60
        print(f"Finished in time: {runtime} minutes")
    else:
        print(f"Finished in time: {runtime} seconds")


def _find_gll_centroids(gll_model, dimensions):
    """
    A function to find the centroid coordinate of gll model
    :param gll: gll model object
    :param dimensions: 1, 2 or 3 dimensions
    :return: array with 3 coordinates per element
    """
    import numpy as np
    import h5py
    with h5py.File(gll_model, 'r') as gll:
        nelements = len(gll['MODEL']['coordinates'][:, 0, 0])

        if dimensions != len(gll['MODEL']['coordinates'][0, 0, :]):
            raise ValueError("Dimensions of GLL model not the same as input")
        centroids = np.zeros(shape=[nelements, dimensions])

        for d in range(dimensions):
            centroids[:, d] = np.mean(gll['MODEL']['coordinates'][:, :, d], axis=1, dtype=np.float64)

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
    import salvus_fem
    import numpy as np
    import h5py
    with h5py.File(gll_model, 'r') as gll:
        for element in nearest_elements:
            print(f"Element: {element}")
            gll_points = gll['MODEL']['coordinates'][element, :, :].T
            # gll_points = np.array(gll_points, order="F")
            gll_points = np.asfortranarray(gll_points)
            point = np.asfortranarray(point)
            print(f"Point: {point}")
            # print(gll_points.flags)
            # print(point.flags)
            # print(f"Shape gll: {gll_points.shape}, shape point: {point.shape}")
            # np.require(gll_points, requirements=["F"])
            # point = point.T
            ref_coord = salvus_fem.hypercube.\
                InverseCoordinateTransformWrapper(n=4, d=3, pnt=point,
                                                  ctrlNodes=gll_points)

            inside = True
            print(f"ref coord: {ref_coord}")
            for i in ref_coord:
                if i < -1.2 or i > 1.2:
                    inside = False
                    break

            if inside:
                print("Inside!")
                return element, ref_coord

    raise IndexError("Could not find an element which this points fits into."
                     " Maybe you should add some tolerance")





























