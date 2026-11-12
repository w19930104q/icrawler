import os
import os.path as osp
import tempfile

from .base import BaseStorage


class FileSystem(BaseStorage):
    """Use filesystem as storage backend.

    The id is filename and data is stored as text files or binary files.
    Writes are atomic: data is first written to a temporary file in the same
    directory and then renamed onto the target, so a failed write leaves the
    target (if any) untouched and never produces a half-written file.
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def write(self, id, data):
        filepath = osp.join(self.root_dir, id)
        folder = osp.dirname(filepath)
        if not osp.isdir(folder):
            try:
                os.makedirs(folder)
            except OSError:
                pass
        mode = "w" if isinstance(data, str) else "wb"
        # NamedTemporaryFile in the target directory guarantees os.replace is
        # an atomic same-filesystem rename; the temp name never collides with
        # the numbered names used by the downloader.
        fd, tmp_path = tempfile.mkstemp(prefix=osp.basename(filepath) + ".", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, mode) as fout:
                fout.write(data)
                fout.flush()
                os.fsync(fout.fileno())
            os.replace(tmp_path, filepath)
        except OSError:
            # Do not leave the temporary payload behind on failure.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def exists(self, id):
        return osp.exists(osp.join(self.root_dir, id))

    def index_exists(self, idx):
        """Check whether any file uses the given numeric index, regardless of
        its extension. Numbering is index-based (``000001.png`` and
        ``000001.jpg`` occupy the same slot), so the skip decision cannot key
        on a single concrete filename.
        """
        prefix = f"{idx:06d}."
        return any(name.startswith(prefix) for name in os.listdir(self.root_dir))

    def max_file_idx(self):
        if not osp.isdir(self.root_dir):
            return 0
        max_idx = 0
        for filename in os.listdir(self.root_dir):
            try:
                idx = int(osp.splitext(filename)[0])
            except ValueError:
                continue
            if idx > max_idx:
                max_idx = idx
        return max_idx
