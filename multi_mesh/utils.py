"""
A few functions to help out with specific tasks
"""
import numpy as np
from multi_mesh.io.exodus import Exodus

def get_rot_matrix(angle, x, y, z):
    """
    :param angle: Rotation angle in radians (Right-Hand rule)
    :param x: x-component of rotational vector
    :param y: y-component of rotational vector
    :param z: z-component of rotational vector
    :return: Rotational Matrix
    """
    # Normalize vector.
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x /= norm
    y /= norm
    z /= norm

    # Setup matrix components.
    matrix = np.empty((3, 3))
    matrix[0, 0] = np.cos(angle) + (x ** 2) * (1 - np.cos(angle))
    matrix[1, 0] = z * np.sin(angle) + x * y * (1 - np.cos(angle))
    matrix[2, 0] = (-1) * y * np.sin(angle) + x * z * (1 - np.cos(angle))
    matrix[0, 1] = x * y * (1 - np.cos(angle)) - z * np.sin(angle)
    matrix[1, 1] = np.cos(angle) + (y ** 2) * (1 - np.cos(angle))
    matrix[2, 1] = x * np.sin(angle) + y * z * (1 - np.cos(angle))
    matrix[0, 2] = y * np.sin(angle) + x * z * (1 - np.cos(angle))
    matrix[1, 2] = (-1) * x * np.sin(angle) + y * z * (1 - np.cos(angle))
    matrix[2, 2] = np.cos(angle) + (z * z) * (1 - np.cos(angle))

    return matrix

def rotate(x, y, z, matrix):
    """

    :param x: x-coordinates to be rotated
    :param y: y-coordinates to be rotated
    :param z: z-coordinates to be rotated
    :param matrix: Rotational matrix obtained from get_rot_matrix
    :return: Rotated x,y,z coordinates
    """

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    return matrix.dot(np.array([x, y, z]))


def rotate_mesh(mesh, event_loc, backwards=False):
    """
    Rotate the coordinates of a mesh to make the source show up below
    the North Pole of the mesh. Can also be used to rotate backwards.
    :param mesh: filename of mesh to be rotated
    :param event_loc: location of event to be rotated to N [lat, lon]
    :param backwards: Backrotation uses transpose of rot matrix
    """
    
    event_vec = [np.cos(event_loc[0]) * np.cos(event_loc[1]),
            np.cos(event_loc[0]) * np.sin(event_loc[1]),
            np.sin(event_loc[0])]
    event_vec = np.normalize(np.array(event_vec))
    north_vec = np.array([0.0, 0.0, 1.0])

    rotate_axis = np.cross(event_vec, north_vec)
    # Make sure that both axis and angle make sense with r-hand-rule
    rot_angle = np.arccos(np.dot(event_vec, north_vec))
    rot_mat = get_rot_matrix(rot_angle, rot_angle[0], rot_angle[1],
            rot_angle[2])
    if backwards:
        rot_mat = rot_mat.T

    mesh = Exodus(mesh)
    rotated_points = rotate(x=mesh.points[:,0], y=mesh.points[:,1],
            z=mesh.points[:,2], matrix=rot_mat)

    mesh.points = rotated_points







