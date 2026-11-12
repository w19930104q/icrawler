import errno
import queue
import threading
import time
from io import BytesIO
from threading import current_thread
from urllib.parse import urlparse

import requests
from PIL import Image

from .utils import ThreadPool

# Outcome labels recorded on every task and aggregated in ``self.stats``.
STATUS_SUCCESS = "success"
STATUS_SKIPPED = "skipped"
STATUS_ABORTED = "aborted"
STATUS_NETWORK_ERROR = "network_error"
STATUS_HTTP_ERROR = "http_error"
STATUS_DECODE_ERROR = "decode_error"
STATUS_REJECTED = "rejected"
STATUS_STORAGE_ERROR = "storage_error"


class Downloader(ThreadPool):
    """Base class for downloader.

    A thread pool of downloader threads, in charge of downloading files and
    saving them in the corresponding paths.

    Quota and filenames are acquired atomically by *successful* downloads.
    A worker fetches and validates a payload without holding any quota; once
    the bytes are known to be good it enters a single critical section that
    checks ``max_num``, hands out a unique monotonically increasing index,
    resolves the filename, writes the file and updates the success counter.
    Any network, decoding or storage failure happens outside the success
    quota, is recorded separately in :attr:`stats` and on the task, and never
    increments ``fetched_num`` or leaves a half-written target file behind.

    Attributes:
        task_queue (CachedQueue): A queue storing image downloading tasks,
            connecting :class:`Parser` and :class:`Downloader`.
        signal (Signal): A Signal object shared by all components.
        session (Session): A session object.
        logger: A logging.Logger object used for logging.
        workers (list): A list of downloader threads.
        thread_num (int): The number of downloader threads.
        lock (Lock): A threading.Lock object guarding the quota bookkeeping.
        storage (BaseStorage): storage backend.
        fetched_num (int): Number of settled slots -- files written plus,
            when ``overwrite`` is False, pre-existing files skipped.
        stats (dict): Counters of every task outcome, keyed by the
            ``STATUS_*`` constants.
    """

    def __init__(self, thread_num, signal, session, storage):
        """Init Parser with some shared variables."""
        super().__init__(thread_num, out_queue=None, name="downloader")
        self.signal = signal
        self.session = session
        self.storage = storage
        self.file_idx_offset = 0
        # Guards both the quota bookkeeping and the in-flight counter. Idle
        # workers wait on this condition instead of exiting while a peer is
        # still processing (e.g. inside a retry loop), so no task is orphaned.
        self._condition = threading.Condition(self.lock)
        self.clear_status()

    def clear_status(self):
        """Reset all quota bookkeeping and outcome counters."""
        self.fetched_num = 0
        # Tasks currently dequeued but not yet settled (download in flight).
        self._in_flight = 0
        # Next never-used *absolute* index. set_file_idx_offset seeds it at the
        # configured offset (``auto`` resolves to the max existing index), so
        # skip probes and filename construction all speak absolute indices and
        # the offset is never added a second time. It only ever grows inside a
        # lock, so every committed file gets a unique index and no number is
        # ever handed out twice or burned by a failure.
        self._next_idx = self.file_idx_offset
        self.stats = {
            STATUS_SUCCESS: 0,
            STATUS_SKIPPED: 0,
            STATUS_ABORTED: 0,
            STATUS_NETWORK_ERROR: 0,
            STATUS_HTTP_ERROR: 0,
            STATUS_DECODE_ERROR: 0,
            STATUS_REJECTED: 0,
            STATUS_STORAGE_ERROR: 0,
        }

    def set_file_idx_offset(self, file_idx_offset=0):
        """Set offset of file index.

        Args:
            file_idx_offset: It can be either an integer or 'auto'. If set
                to an integer, the filename will start from
                ``file_idx_offset`` + 1. If set to ``'auto'``, the filename
                will start from existing max file index plus 1.
        """
        if isinstance(file_idx_offset, int):
            self.file_idx_offset = file_idx_offset
        elif file_idx_offset == "auto":
            self.file_idx_offset = self.storage.max_file_idx()
        else:
            raise ValueError('"file_idx_offset" must be an integer or `auto`')

    def get_filename(self, task, default_ext, file_idx):
        """Build the filename for an atomically assigned absolute index.

        The default strategy is to use an increasing 6-digit number as
        the filename. You can override this method if you want to set custom
        naming rules. The file extension is kept if it can be obtained from
        the url, otherwise ``default_ext`` is used as extension.

        Args:
            task (dict): The task dict got from ``task_queue``.
            file_idx (int): The absolute index atomically assigned to this
                download (already incorporating ``file_idx_offset``).

        Output:
            Filename with extension.
        """
        url_path = urlparse(task["file_url"])[2]
        extension = url_path.split(".")[-1] if "." in url_path else default_ext
        return f"{file_idx:06d}.{extension}"

    def reach_max_num(self):
        """Check if downloaded images reached max num.

        Returns:
            bool: if downloaded images reached max num.
        """
        if self.signal.get("reach_max_num"):
            return True
        if self.max_num > 0 and self.fetched_num >= self.max_num:
            return True
        return False

    def keep_file(self, task, response, **kwargs):
        """Hook for deciding whether an already downloaded file is kept.

        Returning ``False`` rejects the payload: it is never stored and does
        not consume the success quota. The base implementation keeps every
        file; :class:`ImageDownloader` overrides it with image validation.
        """
        return True

    def check_content(self, task, response, **kwargs):
        """Validate the downloaded bytes before they are stored.

        Returns the rejection status (``None`` means the content is accepted).
        The base downloader has no content policy; image validation lives in
        :meth:`ImageDownloader.check_content`.
        """
        if not self.keep_file(task, response, **kwargs):
            return STATUS_REJECTED
        return None

    def _record(self, task, status):
        """Stamp the task and atomically bump one outcome counter."""
        task["status"] = status
        with self.lock:
            self.stats[status] += 1

    def _index_taken(self, idx, filename, overwrite):
        """Whether index ``idx`` is already occupied.

        Prefers the extension-agnostic ``index_exists``; falls back to the
        concrete ``exists`` for storage backends that only implement the base
        interface.
        """
        if overwrite:
            return False
        index_exists = getattr(self.storage, "index_exists", None)
        if callable(index_exists):
            return index_exists(idx)
        return self.storage.exists(filename)

    def _commit(self, task, response, default_ext, overwrite, **kwargs):
        """Atomically take a quota slot, write the file and settle the slot.

        Returns ``True`` only when a new file was durably written. A ``False``
        result leaves ``fetched_num`` untouched and carries the reason in
        ``task['status']`` (skip, abort due to quota, or storage error).
        """
        with self.lock:
            # Re-check under the lock: another thread may have filled the
            # quota while this payload was being fetched.
            if self.max_num > 0 and self.fetched_num >= self.max_num:
                self.signal.set(reach_max_num=True)
                self._record_locked(task, STATUS_ABORTED)
                return False

            # overwrite=False preserves the historical semantics: an occupied
            # index takes a slot (counts towards max_num) but is never fetched
            # again. Walk forward over every occupied index using a task-local
            # variable; the global high-water mark ``_next_idx`` is published
            # once at the end, so one task's skip walk can never corrupt
            # another task's starting index.
            idx = self._next_idx
            filename = self.get_filename(task, default_ext, idx)
            while self._index_taken(idx, filename, overwrite):
                self.fetched_num += 1
                self.stats[STATUS_SKIPPED] += 1
                task["status"] = STATUS_SKIPPED
                if self.max_num > 0 and self.fetched_num >= self.max_num:
                    self.signal.set(reach_max_num=True)
                    # Keep the high-water mark consistent with settled slots.
                    self._next_idx = idx + 1
                    return False
                idx += 1
                filename = self.get_filename(task, default_ext, idx)

            self._next_idx = idx + 1
            try:
                # The storage backend is expected to write atomically (the
                # built-in FileSystem writes a temp file and renames it), so
                # a failure here never leaves a half-written target.
                self.storage.write(filename, response.content)
            except OSError as o:
                # The index was reserved but no target file exists; hand it
                # back by rewinding the high-water mark so numbering stays
                # contiguous for the next successful download.
                self._next_idx = idx
                self.stats[STATUS_STORAGE_ERROR] += 1
                task["status"] = STATUS_STORAGE_ERROR
                self.logger.error("Failed to write file %s, error: %s", filename, o)
                # errno.EINVAL -- name too long
                if o.errno == errno.ENOSPC:
                    self.signal.set(exceed_storage_space=True)
                return False

            self.fetched_num += 1
            self.stats[STATUS_SUCCESS] += 1
            task["success"] = True
            task["filename"] = filename
            task["status"] = STATUS_SUCCESS
            if self.max_num > 0 and self.fetched_num >= self.max_num:
                self.signal.set(reach_max_num=True)
            self.logger.info("image #%s\t%s %s", self.fetched_num, filename, task["file_url"])
            return True

    def _record_locked(self, task, status):
        """Variant of :meth:`_record` for callers already holding the lock."""
        task["status"] = status
        self.stats[status] += 1

    def download(self, task, default_ext, timeout=5, max_retry=3, overwrite=False, **kwargs):
        """Download the file and save it to the corresponding path.

        The network request and content validation consume no success quota.
        Only after the full payload has been accepted does the method enter
        the atomic commit section, which assigns a unique filename and quota
        slot, writes the bytes and updates the success counter. Every other
        outcome is observable via ``task['status']`` and :attr:`stats` and
        never touches ``fetched_num``.

        Args:
            task (dict): The task dict got from ``task_queue``.
            timeout (int): Timeout of making requests for downloading files.
            max_retry (int): the max retry times if the request fails.
            overwrite (bool): Whether to overwrite existing files.

        Returns:
            bool: ``True`` only when a new file was written successfully.
        """
        file_url = task["file_url"]
        task["success"] = False
        task["filename"] = None
        task["status"] = None
        retry = max_retry
        last_status = STATUS_NETWORK_ERROR

        while retry > 0 and not self.signal.get("reach_max_num") and not self.signal.get("exceed_storage_space"):
            try:
                response = self.session.get(file_url, timeout=timeout)
                # Defensive check for sessions that don't raise on HTTP errors.
                if not 200 <= response.status_code < 300:
                    raise requests.HTTPError(f"status code {response.status_code} for {file_url}", response=response)
            except requests.HTTPError as e:
                last_status = STATUS_HTTP_ERROR
                self.logger.error(
                    "HTTP error when downloading file %s, error: %s, remaining retry times: %d",
                    file_url,
                    e,
                    retry - 1,
                )
            except Exception as e:
                last_status = STATUS_NETWORK_ERROR
                self.logger.error(
                    "Exception caught when downloading file %s, error: %s, remaining retry times: %d",
                    file_url,
                    e,
                    retry - 1,
                )
            else:
                # Note: do not raise the global reach_max_num signal here.
                # Several workers may be fetching concurrently; the signal is
                # raised exclusively by _commit once a slot actually settles,
                # so in-flight downloads are never aborted by a peer merely
                # observing the cap. A payload arriving after the cap is
                # filled is rejected atomically inside _commit.
                if self.reach_max_num():
                    self._record(task, STATUS_ABORTED)
                    return False

                rejection = self.check_content(task, response, **kwargs)
                if rejection is not None:
                    # Decoding and policy rejections are final: retrying the
                    # same bytes cannot change the outcome.
                    self._record(task, rejection)
                    return False

                return self._commit(task, response, default_ext, overwrite, **kwargs)
            finally:
                retry -= 1

        # Retries exhausted, or the run was stopped by a global signal.
        if self.signal.get("reach_max_num") or self.signal.get("exceed_storage_space"):
            last_status = STATUS_ABORTED
        self._record(task, last_status)
        return False

    def process_meta(self, task):
        """Process some meta data of the images.

        This method should be overridden by users if wanting to do more things
        other than just downloading the image, such as saving annotations. It
        is only invoked for tasks whose payload was fully downloaded and
        committed to the storage.

        Args:
            task (dict): The task dict got from task_queue. This method will
                make use of fields other than ``file_url`` in the dict.
        """
        pass

    def start(self, file_idx_offset=0, *args, **kwargs):
        # Resolve the offset first, then clear_status seeds the absolute index
        # counter at it; otherwise clear_status would reset _next_idx to 0.
        self.set_file_idx_offset(file_idx_offset)
        self.clear_status()
        self.init_workers(*args, **kwargs)
        for worker in self.workers:
            worker.start()
            self.logger.debug("thread %s started", worker.name)

    def worker_exec(self, max_num, default_ext="", queue_timeout=5, req_timeout=5, max_idle_time=None, **kwargs):
        """Target method of workers.

        Get task from ``task_queue`` and then download files and process meta
        data. A downloader thread will exit in either of the following cases:

        1. All parser threads have exited, the task_queue is empty and no peer
           is still processing a task (in-flight work has fully settled).
        2. Downloaded image number has reached required number(max_num).
        3. No new downloads for max_idle_time seconds.

        Args:
            max_num (int): Maximum number of images to download
            queue_timeout (int): Timeout of getting tasks from ``task_queue``.
            req_timeout (int): Timeout of making requests for downloading pages.
            max_idle_time (int): Maximum time (in seconds) to wait without
                receiving new images
            **kwargs: Arguments passed to the :func:`download` method.
        """
        self.max_num = max_num
        last_download_time = time.monotonic()

        while True:
            if self.signal.get("reach_max_num"):
                self.logger.info("downloaded images reach max num, thread %s is ready to exit", current_thread().name)
                break
            if self.signal.get("exceed_storage_space"):
                self.logger.info("storage space exceeded, thread %s is ready to exit", current_thread().name)
                break

            current_time = time.monotonic()
            if max_idle_time is not None and current_time - last_download_time > max_idle_time and self.fetched_num > 0:
                self.logger.info("no new images for %d seconds, thread %s exit", max_idle_time, current_thread().name)
                break

            try:
                task = self.in_queue.get(timeout=queue_timeout)
            except queue.Empty:
                # Re-check global stop signals first: a worker blocked in
                # get() must wake promptly once the cap is reached or storage
                # is exhausted.
                if self.signal.get("reach_max_num") or self.signal.get("exceed_storage_space"):
                    break
                with self._condition:
                    # Only shut down when parsing is finished, nothing remains
                    # queued and every dequeued task has settled. Otherwise keep
                    # waiting rather than abandoning tasks still inside a
                    # download or retry loop.
                    if self.signal.get("parser_exited") and self._in_flight == 0 and self.in_queue.empty():
                        self.logger.info("no more download task for thread %s", current_thread().name)
                        break
                    if self.fetched_num == 0 and not self.signal.get("parser_exited"):
                        self.logger.info("%s is waiting for new download tasks", current_thread().name)
                    self._condition.wait(timeout=queue_timeout)
                continue
            except Exception as e:
                self.logger.error("exception in thread %s: %s", current_thread().name, e)
                continue

            with self._condition:
                self._in_flight += 1
            try:
                success = self.download(task, default_ext, req_timeout, **kwargs)
                if success:
                    last_download_time = time.monotonic()
                    # process_meta only runs for a fully committed file.
                    self.process_meta(task)
            finally:
                self.in_queue.task_done()
                with self._condition:
                    self._in_flight -= 1
                    self._condition.notify_all()

        self.logger.info(
            "thread %s exit, settled files: %d, stats: %s",
            current_thread().name,
            self.fetched_num,
            self.stats,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("all downloader threads exited")


class ImageDownloader(Downloader):
    """Downloader specified for images."""

    # Extensions accepted directly from the URL path; anything else falls
    # back to the caller-provided default extension.
    VALID_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "tiff", "gif", "ppm", "pgm")

    def _size_lt(self, sz1, sz2):
        return max(sz1) <= max(sz2) and min(sz1) <= min(sz2)

    def _size_gt(self, sz1, sz2):
        return max(sz1) >= max(sz2) and min(sz1) >= min(sz2)

    def keep_file(self, task, response, min_size=None, max_size=None):
        """Decide whether to keep the image.

        The image is fully decoded (not merely sniffed) so truncated or
        otherwise undecodable payloads are rejected before anything touches
        the storage, and its size is compared against ``min_size`` and
        ``max_size``.

        Args:
            response (Response): response of requests.
            min_size (tuple or None): minimum size of required images.
            max_size (tuple or None): maximum size of required images.
        Returns:
            bool: whether to keep the image.
        """
        try:
            img = Image.open(BytesIO(response.content))
            # Force the whole payload to be decoded: Image.open only reads the
            # header and would otherwise accept a truncated image.
            img.load()
        except OSError:
            task["_reject_status"] = STATUS_DECODE_ERROR
            return False
        task["img_size"] = img.size
        if min_size and not self._size_gt(img.size, min_size):
            task["_reject_status"] = STATUS_REJECTED
            return False
        if max_size and not self._size_lt(img.size, max_size):
            task["_reject_status"] = STATUS_REJECTED
            return False
        task.pop("_reject_status", None)
        return True

    def check_content(self, task, response, **kwargs):
        """Image-specific content validation with distinct failure labels."""
        task.pop("_reject_status", None)
        if not self.keep_file(task, response, **kwargs):
            return task.pop("_reject_status", STATUS_REJECTED)
        return None

    def get_filename(self, task, default_ext, file_idx):
        url_path = urlparse(task["file_url"])[2]
        if "." in url_path:
            extension = url_path.split(".")[-1]
            if extension.lower() not in self.VALID_EXTENSIONS:
                extension = default_ext
        else:
            extension = default_ext
        return f"{file_idx:06d}.{extension}"

    def worker_exec(self, max_num, default_ext="jpg", queue_timeout=5, req_timeout=5, max_idle_time=None, **kwargs):
        super().worker_exec(max_num, default_ext, queue_timeout, req_timeout, max_idle_time, **kwargs)
