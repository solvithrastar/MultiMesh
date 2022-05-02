import h5py
import numpy as np
import pathlib
from typing import Union


class SalvusMesh(object):
    """
    Class which is designed to read the useful features of a Salvus mesh
    and do so faster and more reliably than UnstructuredMesh.from_h5()
    """

    def __init__(self, filename: Union[str, pathlib.Path], fast_mode: bool = True):
        """
        Opens the h5py file and reads a few mandatory values into memory
        and there are also two optional dictionaries to build.

        :param filename: [description]
        :type filename: Union[str, pathlib.Path]
        :param fast_mode: If True, will not load fields into dictionaries,
            defaults to True
        :type fast_mode: bool, optional
        """
        self.filename = filename
        with h5py.File(self.filename, "r") as self.dataset:
            self.points = self.get_points()
            self.nelem = self.get_nelem()
            self.n_gll_points = self.get_n_gll_points()
            self.dimensions = self.get_dimensions()
            self.shape_order = self.get_shape_order()
            self.global_strings = self.get_global_strings()
            self.elemental_parameter_indices = self.get_elemental_parameter_indices()
            self.nodal_parameter_indices = self.get_nodal_parameter_indices()
            if not fast_mode:
                self.elemental_fields = self.get_elemental_fields()
                self.element_nodal_fields = self.get_element_nodal_fields()

    def get_points(self):
        return self.dataset["MODEL/coordinates"][()]

    def get_n_gll_points(self):
        return self.points.shape[1]

    def get_dimensions(self):
        return self.points.shape[2]

    def get_shape_order(self):
        return int(np.round(self.n_gll_points ** (1 / self.dimensions)) - 1)

    def get_nelem(self):
        return self.points.shape[0]

    def get_global_strings(self):
        global_str = {}
        for key, val in self.dataset["MODEL"].attrs.items():
            if isinstance(val, np.bytes_):
                global_str[key] = val
        return global_str

    def update_global_strings(self, dataset):
        global_str = {}
        for key, val in dataset["MODEL"].attrs.items():
            if isinstance(val, np.bytes_):
                global_str[key] = val
        self.global_strings = global_str

    def get_nodal_parameter_indices(self):
        indices = self.dataset["MODEL/data"].attrs.get("DIMENSION_LABELS")[1]
        if not type(indices) == str:
            indices = indices.decode()
        indices = indices.replace(" ", "")[1:-1].split("|")
        return indices

    def get_elemental_parameter_indices(self):
        indices = self.dataset["MODEL/element_data"].attrs.get("DIMENSION_LABELS")[1]
        if not type(indices) == str:
            indices = indices.decode()
        indices = indices.replace(" ", "")[1:-1].split("|")
        return indices

    def get_elemental_fields(self):
        if f"{self}.elemental_fields" in locals():
            return self.elemental_fields
        e_fields = {}
        for _i, param in enumerate(self.elemental_parameter_indices):
            with h5py.File(self.filename, "r") as dataset:
                e_fields[param] = dataset["MODEL/element_data"][:, _i]
        return e_fields

    def get_element_nodal_fields(self):
        if "self.element_nodal_fields" in locals():
            return self.element_nodal_fields
        e_n_fields = {}
        for _i, param in enumerate(self.nodal_parameter_indices):
            with h5py.File(self.filename, "r") as dataset:
                e_n_fields[param] = dataset["MODEL/data"][:, _i, :]
        return e_n_fields

    def get_element_centroids(self):
        return np.mean(self.points, axis=1)

    def get_element_nodes(self):
        return self.points

    def get_element_nodal_field(self, param):
        ind = self.get_nodal_parameter_indices().index(param)
        with h5py.File(self.filename, "r") as ds:
            return ds["MODEL/data"][:, ind, :]

    def get_elemental_field(self, param):
        ind = self.get_elemental_parameter_indices().index(param)
        with h5py.File(self.filename, "r") as ds:
            return ds["MODEL/data"][:, ind]

    def set_global_string(
        self,
        name: str,
        value: str,
    ):
        """
        Loads the actual hdf5 dataset and writes the new value into the right
        place
        """
        assert isinstance(value, str), "Value needs to be a string"
        assert isinstance(name, str), "Name needs to be a string"
        with h5py.File(self.filename, "r+") as ds:
            if name in self.global_strings.keys():
                ds["MODEL"].attrs.modify(
                    name=name, value=np.array([value], dtype=np.bytes_)
                )
            else:

                ds["MODEL"].attrs.create(name=name, data=value, dtype=np.bytes_)
            self.update_global_strings(ds)

    def attach_field(
        self,
        name: str,
        data: np.ndarray,
    ):
        """
        Attach either an elemental field or an elemental nodal field
        to the mesh. This does currently not update the fields which
        is associated with this object. That can be arranged but
        would take time so don't know if desirable.
        If you want to reload it you can always recreate the object.

        :param name: Name of field
        :type name: str
        :param data: Values for field
        :type data: np.ndarray
        """
        nodal_field = False
        elemental_field = False
        assert isinstance(data, np.ndarray), "Data needs to be a numpy array"
        if data.shape == (self.nelem, self.n_gll_points):
            nodal_field = True
        elif data.shape == (self.nelem):
            elemental_field = True
        else:
            raise ValueError(
                "We can only attach elemental_nodal_field or elemental_fields"
            )
        with h5py.File(self.filename, "r+") as ds:
            if nodal_field:
                if name in self.nodal_parameter_indices:
                    ind = self.nodal_parameter_indices.index(name)
                    ds["MODEL/data"][:, ind, :] = data
                    print(f"Attached field {name} to mesh")
                else:
                    raise ValueError("Currently we only attach existing fields")
            if elemental_field:
                if name in self.elemental_parameter_indices:
                    ind = self.elemental_parameter_indices.index(name)
                    ds["MODEL/element_data"][:, ind] = data
                    print(f"Attached elemental field {name} to mesh")
                else:
                    raise ValueError("Currently we only attach existing fields")
