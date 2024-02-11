"""Generic checkpointing objects relying on the ``h5py`` library. Handles any
number of variables to be saved, but only expects ``torch.Tensor``, ``int`` or
variables describing the state of a ``torch._C.Generator`` object.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# Code from this file is directly adapted from the Python code
# available https://gitlab.cristal.univ-lille.fr/pthouven/dsgs (GPL 3.0
# license), associated with the folllowing paper
#
# P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

from abc import ABC, abstractmethod

import h5py
import numpy as np
import torch

# TODO: add "device" to explicitly state where to load variables?
# TODO- (requiring a list?)
# ! reformat load functionality (put open / close methods to use properly to
# ! save the elements, or pass type to the checkpointer, len/shape, ...)

# ! clean save functionality (messy at the moment)


class BaseCheckpoint(ABC):
    r"""Base checkpoint object gathering the parameters common to the
    checkpoint schemes used in this library.

    .. _hdf5plugin: http://www.silx.org/doc/hdf5plugin/latest/usage.html#hdf5plugin.Blosc

    Attributes
    ----------
    root_filename : str
        Root of the filename (containing path to the appropriate directory)
        where the checkpoint file is / will be stored.
    cname : str
        Name of the hdf5 compression filter (aka compressor). Default to
        "gzip".
    clevel : int
        Compression level. Default to 5 (default for Blosc).
    shuffle : int
        Byte shuffle option (see `hdf5plugin`_ documentation). Default to 1.
        Not used for the moment.

    Note
    ----
        The following virtual methods need to be implemented in any daughter class:

        - :meth:`checkpoint.BaseCheckpoint.save`,
        - :meth:`checkpoint.BaseCheckpoint.load`.
    """

    def __init__(
        self,
        root_filename,
        cname="gzip",
        clevel=5,
        shuffle=1,
    ):
        """
        Parameters
        ----------
        root_filename : str
            Root of the filename (containing path to the appropriate directory)
            where the checkpoint file is / will be stored.
        cname : str
            Name of the hdf5 compression filter (aka compressor). Default to
            "gzip".
        clevel : int
            Compression level. Default to 5 (default for Blosc).
        shuffle : int
            Byte shuffle option (see hdf5plugin_ documentation). Default to 1.
            Not used for the moment.
        """
        self.root_filename = root_filename
        self.cname = cname
        self.clevel = clevel
        self.shuffle = shuffle

    def filename(self, file_id):
        """Get name of target file.

        Parameters
        ----------
        file_id : str or int
            String or integer describing the id of the target ``.h5`` file.

        Returns
        -------
        str
            Target filename.
        """
        return "{}{}.h5".format(self.root_filename, file_id)

    @abstractmethod
    def save(
        self,
        file_id,
        chunk_sizes,
        dtypes,
        rng=None,
        mode="w",
        rdcc_nbytes=None,
        **kwargs,
    ):  # pragma: no cover
        r"""Saving content of the input variables within the dictionary
        kwargs to disk.

        Parameters
        ----------
        file_id : str or int
            String or integer describing the id of the target ``.h5`` file.
        chunk_sizes : list of tuples, of length ``n``
            List of tuples representing the chunk size for each input variable.
            For scalar input, the corresponding chunk size needs to be
            ``None``.
        dtypes : list[type]
            List of type, corresponding to the different variables to save.
        rng : torch._C.Generator or None, optional
            Random number generator to be restored using specific state stored
            on disk, by default None.
        mode : str, optional
            Mode to open the h5 file ("a", or "w"). By default "w".
        rdcc_nbytes : float, optional
            Sets the total size (measured in bytes) of the raw data chunk cache
            for each dataset. The default size is 1 MB. This should be set to
            the size of each chunk times the number of chunks that are likely
            to be needed in cache. By default None (see
            `h5py documentation <https://docs.h5py.org/en/stable/high/file.html>`_).
        kwargs : list
            List of keyword arguments reprensenting Python variables to be
            saved.
        """
        pass

    @abstractmethod
    def load(self, file_id, select, *args, rng=None):  # pragma: no cover
        r"""Loading some variables from a checkpoint file.

        Parameters
        ----------
        file_id : str or int
            String or integer describing the id of the target ``.h5`` file.
        select : list of slices, of size ``n``
            List of slices to load part of the corresponding variable from disk.
        args : list of str, of size ``n``
            Variable list of strings corresponding to the name of the variables
            to be loaded.
        rng : torch._C.Generator or None, optional
            Random number generator to be restored using specific state stored
            on disk. By default None.

        Returns
        -------
        NotImplemented
        """
        return NotImplemented


class SerialCheckpoint(BaseCheckpoint):
    r"""Checkpoint in serial environments (i.e., without MPI), using ``h5py``.

    Attributes
    ----------
    root_filename : str
        Root of the filename (containing path to the appropriate directory)
        where the checkpoint file is / will be stored.
    cname : str
        Name of the hdf5 compression filter (aka compressor). Default to
        "gzip".
    clevel : int
        Compression level. Default to 5 (default for Blosc).
    shuffle : int
        Byte shuffle option (see hdf5plugin_ documentation). Default to 1.
        Not used for the moment.

    .. _hdf5plugin: http://www.silx.org/doc/hdf5plugin/latest/usage.html#hdf5plugin.Blosc
    """

    def __init__(
        self,
        root_filename,
        cname="gzip",
        clevel=5,
        shuffle=1,
    ):
        """
        Parameters
        ----------
        root_filename : str
            Root of the filename (containing path to the appropriate directory)
            where the checkpoint file is / will be stored.
        cname : str
            Name of the hdf5 compression filter (aka compressor). Default to
            "gzip".
        clevel : int
            Compression level. Default to 5 (default for Blosc).
        shuffle : int
            Byte shuffle option (see hdf5plugin_ documentation). Default to 1.
            Not used for the moment.

        .. _hdf5plugin: http://www.silx.org/doc/hdf5plugin/latest/usage.html#hdf5plugin.Blosc
        """
        super(SerialCheckpoint, self).__init__(root_filename, cname, clevel, shuffle)

    def save(
        self,
        file_id,
        chunk_sizes,
        dtypes,
        rng=None,
        mode="w",
        rdcc_nbytes=None,
        **kwargs,
    ):
        r"""Saving content of the input variables within the dictionary
        kwargs to disk.

        Parameters
        ----------
        file_id : str or int
            String or integer describing the id of the target ``.h5`` file.
        chunk_sizes : list of tuples, of length ``n``
            List of tuples representing the chunk size for each input variable.
            For scalar input, the corresponding chunk size needs to be
            ``None``.
        dtypes : list[type]
            List of type, corresponding to the different variables to save.
        rng : torch._C.Generator or None, optional
            Save to disk the current state of a random number generator, if
            any. By default None.
        mode : str, optional
            Mode to open the h5 file ("a", or "w"). By default "w".
        rdcc_nbytes : float, optional
            Sets the total size (measured in bytes) of the raw data chunk cache
            for each dataset. The default size is 1 MB. This should be set to
            the size of each chunk times the number of chunks that are likely
            to be needed in cache. By default None (see
            `h5py documentation <https://docs.h5py.org/en/stable/high/file.html>`_).
        kwargs : list
            List of keyword arguments reprensenting Python variables to be
            saved.

        Example
        -------
        filename = "test"
        file_id = 1
        chkpt = SerialCheckpoint(filename)

        # data to be saved
        rng = torch.manual_seed(1234)
        a = torch.normal(0.0, 1.0, (3,), generator=rng)

        # saving "d" with state of the random number generator into "test.h5"
        chunk_sizes = [None]  # no chunking size selected for .h5
        chkpt.save(file_id, chunk_sizes, rng=rng, a=a)
        """
        filename_ = self.filename(file_id)

        with h5py.File(filename_, mode, rdcc_nbytes=rdcc_nbytes) as f:
            # * backup state random number generator
            if rng is not None:
                current_state = rng.get_state()
                dset = f.create_dataset(
                    "rng_state", current_state.shape, dtype=int
                )  # (5056,), torch.uint8
                dset[:] = current_state

            # * backup other variables
            for count, (var_name, var) in enumerate(kwargs.items()):
                if dtypes[count] == torch.float32:
                    type_ = np.float32
                else:
                    type_ = dtypes[count]

                if isinstance(var, torch.Tensor):
                    if var.numel() > 1:
                        dset = f.create_dataset(
                            var_name,
                            var.shape,
                            dtype=type_,
                            compression=self.cname,
                            compression_opts=self.clevel,
                            chunks=chunk_sizes[count],
                        )
                    else:
                        dset = f.create_dataset(var_name, (1,), dtype=type_)
                elif isinstance(var, torch.Size):
                    dset = f.create_dataset(var_name, (len(var),), dtype="i")
                else:
                    # ! when input is a scalar, only allow integer type
                    dset = f.create_dataset(var_name, (1,), dtype="i")
                dset[()] = var

    def load(self, file_id, select, *args, rng=None, rdcc_nbytes=None):
        r"""Loading some variables from a checkpoint file.

        Parameters
        ----------
        file_id : str or int
            String or integer describing the id of the target ``.h5`` file.
        select : list of slices, of size ``n``
            List of slices to load part of the corresponding variable from disk.
        args : list of str, of size ``n``
            Variable list of strings corresponding to the name of the variables
            to be loaded.
        rng : torch._C.Generator or None, optional
            Random number generator to be restored using specific state stored
            on disk. By default None.
        rdcc_nbytes : float, optional
            Sets the total size (measured in bytes) of the raw data chunk cache
            for each dataset. The default size is 1 MB. This should be set to
            the size of each chunk times the number of chunks that are likely
            to be needed in cache. By default None (see
            `h5py documentation <https://docs.h5py.org/en/stable/high/file.html>`_).

        Returns
        -------
        dic_var : dict
            Dictionary containing all the variables loaded from the ``.h5``
            file.

        Raises
        ------
        TypeError
            Unsupported data type.

        Example
        -------
        >>>> filename = "test"
        >>>> file_id = 1
        >>>> chkpt = SerialCheckpoint(filename)
        >>>> rng = torch.manual_seed()

        >>>> # restore state of "rng" and load "a[1:, 1:]" from "test.h5"
        >>>> select = [None]
        >>>> chkpt.load(file_id, [(slice(1, None, None), slice(1, None, None))],"a", rng=rng)
        >>>> chkpt.save(file_id, select, "a", rng=rng)
        """
        filename_ = self.filename(file_id)
        dic_var = {}

        with h5py.File(filename_, "r", rdcc_nbytes=rdcc_nbytes) as f:
            if rng is not None:
                loaded_state = torch.tensor(f["rng_state"][:], dtype=torch.uint8)
                rng.set_state(loaded_state)

            for count, var in enumerate(args):
                if f[var].dtype.__str__() == "int32":
                    dic_var[var] = torch.tensor(
                        f[var][select[count]], dtype=torch.int32
                    )
                elif f[var].dtype.__str__() == "int64":
                    dic_var[var] = torch.tensor(
                        f[var][select[count]], dtype=torch.int64
                    )
                elif f[var].dtype.__str__() == "complex64":
                    dic_var[var] = torch.tensor(
                        f[var][select[count]], dtype=torch.complex64
                    )
                elif f[var].dtype.__str__() == "float32":
                    dic_var[var] = torch.tensor(
                        f[var][select[count]], dtype=torch.float32
                    )
                elif f[var].dtype.__str__() == "bool":
                    dic_var[var] = torch.tensor(f[var][select[count]], dtype=bool)
                else:
                    raise TypeError("Unsupported data type.")

        return dic_var


if __name__ == "__main__":
    # TODO: turn example into a proper unit-test

    # * example use serial checkpoint class
    filename = "test"
    file_id = 1
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    chkpt = SerialCheckpoint(filename)

    # data to be saved
    # rng = torch.manual_seed(1234)  # on cpu by default
    rng = torch.Generator(device=device)
    rng.manual_seed(1234)
    c = torch.ones((2, 2))
    d = torch.normal(0.0, 1.0, (3,), generator=rng)
    complex_array = (1 + 1j) * torch.ones((2,))

    # saving
    chunk_sizes = 5 * [None]
    chkpt.save(
        file_id,
        chunk_sizes,
        [int, int, float, float, complex],
        rng=rng,
        a=3,
        b=4,
        c=c,
        d=d,
        complex_array=complex_array,
    )
    e = torch.normal(0.0, 1.0, (3,), generator=rng)

    # loading complex-valued array
    dict_complex = chkpt.load(file_id, [slice(None, None, None)], "complex_array")

    # restoring rng and loading values from disk
    dic_var = chkpt.load(
        1, 4 * [slice(None, None, None)], "a", "b", "c", "d", rng=rng
    )  # np.s_[:]

    # check consistency
    consistency = np.allclose(3, dic_var["a"][0])
    f = torch.normal(0.0, 1.0, (3,), generator=rng)
    print("Consistency a after loading?: {}".format(consistency))
    print("Consistency rng?: {}".format(torch.allclose(e, f)))
    print("Consistency loading?: {}".format(torch.allclose(d, dic_var["d"])))

    # * loading only a slice of a variable
    # s = slice(5, None, 4)
    # print(s.start, s.stop, s.step)
    dic_sliced = chkpt.load(
        file_id, [(slice(1, None, None), slice(1, None, None))], "c", rng=None
    )  # loading only c[1:, 1:]
    consistency = np.allclose(c[1:, 1:], dic_sliced["c"])
    print("Consistency sliced b after loading?: {}".format(consistency))

    pass
