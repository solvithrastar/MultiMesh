import multiprocessing
import numpy as np
from multi_mesh.components.interpolator import inverse_transform
from multi_mesh.components.interpolator import get_coefficients
from pykdtree.kdtree import KDTree
from tqdm import tqdm


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
    """Takes a salvus mesh and converts it to a sphere.
    Acts on the passed object
    """

    _, i = np.unique(mesh.connectivity, return_index=True)
    rad_1D = mesh.element_nodal_fields["z_node_1D"].flatten()[i]

    r_earth = 6371000
    x, y, z = mesh.points.T
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Conert all points that do not lie right in the core
    x[r > 0] = x[r > 0] * r_earth * rad_1D[r > 0] / r[r > 0]
    y[r > 0] = y[r > 0] * r_earth * rad_1D[r > 0] / r[r > 0]
    z[r > 0] = z[r > 0] * r_earth * rad_1D[r > 0] / r[r > 0]


def get_element_weights(gll_points, centroid_tree, points):
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

                if np.any(np.isnan(ref_coord)):
                    continue

                # tolerance of 5%
                if np.all(np.abs(ref_coord) < 1.05):
                    coeffs = get_coefficients(
                        2,
                        0,
                        0,
                        np.asfortranarray(ref_coord, dtype=np.float64),
                        3,
                    )
                    return element, coeffs
            # return weights zero if nothing found
            return -1, np.zeros(27)

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


def interpolate_to_points(mesh, points, params_to_interp,
                          make_spherical=False, centroid_tree=None):

    """
    Interpolates from a mesh to point cloud.

    :param mesh: Mesh from which you want to interpolate
    :param points: np.array of points that require interpolation,
    if they are not found. zero is returned
    :param params_to_interp: list of params to interp
    :param make_spherical: bool that determines if mesh gets mapped to a sphere.
    Careful. Setting this will alter the passed object.
    :param centroid_tree: KDTree initialized from the centroids of the elements
    of mesh. Passing this is optional,, but helps to speed up this
    function when it is placed in a loop.
    :return: array[nparams_to_interp, npoints]
    """

    if make_spherical:
        map_to_sphere(mesh)

    if not centroid_tree:
        print("Initializing KDtree...")
        elem_centroid = mesh.get_element_centroid()
        centroid_tree = KDTree(elem_centroid)

    # Get GLL points from old mesh
    gll_points = mesh.points[mesh.connectivity]

    # Get elements and interpolation coefficients for new_points
    print("Retrieving interpolation weigts")
    elem_indices, coeffs = get_element_weights(
        gll_points, centroid_tree, points
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


def interpolate_to_mesh(
    old_mesh, new_mesh, params_to_interp=["VSV", "VSH", "VPV", "VPH"]
):
    """
    Maps both meshes to a sphere and interpolate values
    from old mesh to new mesh for params to interp.
    Returns the original coordinate system

    Values that are not found are given zero
    """
    # store original point locations
    orig_old_elliptic_mesh_points = np.copy(old_mesh.points)
    orig_new_elliptic_mesh_points = np.copy(new_mesh.points)

    # Map both meshes to a sphere
    map_to_sphere(old_mesh)
    map_to_sphere(new_mesh)
    vals = interpolate_to_points(old_mesh, new_mesh.points, params_to_interp)

    for i, param in enumerate(params_to_interp):
        new_element_nodal_vals = vals[:, i][new_mesh.connectivity]
        new_mesh.element_nodal_fields[param][:] = new_element_nodal_vals

    # Restore original coordinates
    old_mesh.points = orig_old_elliptic_mesh_points
    new_mesh.points = orig_new_elliptic_mesh_points
