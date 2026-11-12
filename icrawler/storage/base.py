from abc import ABCMeta, abstractmethod


class BaseStorage:
    """Base class of backend storage"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def write(self, id, data):
        """Abstract interface of writing data

        Args:
            id (str): unique id of the data in the storage.
            data (bytes or str): data to be stored.
        """
        return

    @abstractmethod
    def exists(self, id):
        """Check the existence of some data

        Args:
            id (str): unique id of the data in the storage

        Returns:
            bool: whether the data exists
        """
        return False

    def index_exists(self, idx):
        """Check whether a numeric index is already occupied.

        Filenames are derived from numeric indices but can carry different
        extensions (e.g. ``000001.png`` and ``000001.jpg`` share one slot), so
        deciding whether an index is free cannot rely on a single id. The
        default implementation is conservative and reports the index as free;
        backends that enumerate their objects (the built-in FileSystem)
        override it with an exact, extension-agnostic check.

        Args:
            idx (int): the numeric file index to probe.

        Returns:
            bool: whether any stored object already uses the index.
        """
        return False

    @abstractmethod
    def max_file_idx(self):
        """Get the max existing file index

        Returns:
            int: the max index
        """
        return 0
