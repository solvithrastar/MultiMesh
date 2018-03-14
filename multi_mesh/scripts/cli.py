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
