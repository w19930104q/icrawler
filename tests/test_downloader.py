"""Regression tests for the atomic-quota ImageDownloader.

The tests run fully offline against either a local HTTP server
(:class:`http.server.ThreadingHTTPServer`) serving real images from a temp
directory, or lightweight fakes injected into the downloader. Every artifact
lands in a per-test temporary directory.
"""

import io
import os
import os.path as osp
import queue as queue_mod
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from PIL import Image

from icrawler.downloader import (
    STATUS_DECODE_ERROR,
    STATUS_HTTP_ERROR,
    STATUS_NETWORK_ERROR,
    STATUS_REJECTED,
    STATUS_SKIPPED,
    STATUS_STORAGE_ERROR,
    STATUS_SUCCESS,
    ImageDownloader,
)
from icrawler.storage.filesystem import FileSystem
from icrawler.utils import Signal


# ---------------------------------------------------------------------------
# Test fixtures: real images, a local HTTP server and fakes
# ---------------------------------------------------------------------------
def make_image(color, size=(32, 24), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


GOOD_PNG = make_image((255, 0, 0))
GOOD_JPG = make_image((0, 255, 0), size=(48, 48), fmt="JPEG")
BIG_PNG = make_image((0, 0, 255), size=(200, 200))
NOT_AN_IMAGE = b"this is definitely not an image payload"
# A valid header followed by a truncated payload: Image.open succeeds but a
# full decode (img.load()) must fail.
TRUNCATED_PNG = GOOD_PNG[: len(GOOD_PNG) // 2]


class ImageHandler(BaseHTTPRequestHandler):
    """Serves byte payloads keyed by path; supports flaky and error routes.

    Routes and attempt counters live on the server instance, so concurrent
    servers in the same test session never share state.
    """

    def log_message(self, *args):  # silence stderr noise
        pass

    # Alias prefix -> concrete route, so many distinct URLs can share payloads.
    aliases = {"r": "red.png", "g": "green.jpg", "b": "bad.png", "m": "missing.png"}

    def do_GET(self):
        path = self.path.lstrip("/")
        spec = self.server.routes.get(path)
        if spec is None:
            # Resolve numbered aliases such as "r07.png" to a shared route.
            stem = path.split(".")[0]
            prefix = stem.rstrip("0123456789")
            if stem != prefix and prefix in self.aliases:
                spec = self.server.routes.get(self.aliases[prefix])
        if spec is None:
            self.send_response(404)
            self.end_headers()
            return
        if isinstance(spec, dict):
            data = spec["data"]
            flaky = spec.get("flaky", 0)
            status = spec.get("status", 200)
        elif isinstance(spec, tuple):
            # Either (data, status) or (data, flaky, status).
            data = spec[0]
            if len(spec) == 2:
                flaky, status = 0, spec[1]
            else:
                _, flaky, status = spec
        else:
            data, flaky, status = spec, 0, 200
        attempts = self.server.attempts
        count = attempts[path] = attempts.get(path, 0) + 1
        if count <= flaky:
            # Fail the first `flaky` attempts with a 500, then serve the bytes.
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)


class LocalServer:
    def __init__(self, routes):
        self.attempts = {}
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), ImageHandler)
        self.httpd.routes = routes
        self.httpd.attempts = self.attempts
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self):
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def url(self, name):
        return f"{self.base}/{name}"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def server():
    routes = {
        "red.png": GOOD_PNG,
        "green.jpg": GOOD_JPG,
        "blue.png": BIG_PNG,
        "bad.png": NOT_AN_IMAGE,
        "truncated.png": TRUNCATED_PNG,
        "missing.png": (b"", 0, 404),
        "flaky.png": {"data": GOOD_PNG, "flaky": 2},
    }
    srv = LocalServer(routes).start()
    yield srv
    srv.stop()


class FakeResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code


class FakeSession:
    """Session stub: scripted responses/exceptions, records every call."""

    def __init__(self, scripted):
        # scripted: url -> list of either Exception instances or (content, status)
        self.scripted = scripted
        self.calls = []

    def get(self, url, timeout=5):
        self.calls.append((url, timeout))
        behavior = self.scripted[url].pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        content, status = behavior
        return FakeResponse(content, status)


class FailingStorage(FileSystem):
    """Storage whose write raises OSError a scripted number of times.

    Because the production FileSystem write is atomic (temp file + rename),
    a failure leaves neither a target file nor a temp file behind.
    """

    def __init__(self, root_dir, fail_times=1, err=None):
        super().__init__(root_dir)
        self.remaining = fail_times
        self.err = err if err is not None else OSError("simulated write failure")
        self.write_calls = 0

    def write(self, id, data):
        self.write_calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.err
        super().write(id, data)


# ---------------------------------------------------------------------------
# Helpers to drive a downloader directly, without Feeder/Parser
# ---------------------------------------------------------------------------
def build_downloader(tmp_path, thread_num=4, session=None, storage=None, log_level="WARNING"):
    storage = storage or FileSystem(str(tmp_path))
    if session is None:
        # Plain requests session: the downloader owns retry policy, so failed
        # status codes return immediately instead of triggering tenacity's
        # exponential backoff.
        session = requests.Session()
    dl = ImageDownloader(
        thread_num=thread_num,
        signal=Signal(),
        session=session,
        storage=storage,
    )
    dl.signal.set(feeder_exited=False, parser_exited=False, reach_max_num=False, exceed_storage_space=False)
    dl.logger.setLevel(log_level)
    return dl


def enqueue(dl, urls):
    for url in urls:
        dl.in_queue.put({"file_url": url})


def run(dl, max_num=0, overwrite=False, file_idx_offset=0, max_idle_time=None, drain=True, **download_kwargs):
    """Run worker threads and join them.

    With ``drain=True`` (the default), ``parser_exited`` is kept False until
    every queued task has been dequeued and settled, mirroring
    :class:`~icrawler.crawler.Crawler`, which only raises the signal once the
    parser threads have actually finished. Combined with the downloader's
    in-flight guard, no task is orphaned inside a retry loop. With
    ``drain=False`` the parser is marked exited immediately, modelling a run
    whose queue still holds surplus tasks when ``max_num`` is reached.
    """
    dl.signal.set(parser_exited=not drain)
    kwargs = dict(
        max_num=max_num,
        default_ext="jpg",
        queue_timeout=2,
        req_timeout=5,
        max_retry=3,
        overwrite=overwrite,
        max_idle_time=max_idle_time,
    )
    kwargs.update(download_kwargs)
    dl.start(file_idx_offset=file_idx_offset, **kwargs)
    if drain:
        deadline = time.monotonic() + 60
        while dl.in_queue.unfinished_tasks > 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert dl.in_queue.unfinished_tasks == 0, "tasks were abandoned before settling"
        dl.signal.set(parser_exited=True)
        # Wake any worker blocked in queue.get() so it observes the signal.
        with dl._condition:
            dl._condition.notify_all()
    for worker in dl.workers:
        worker.join(timeout=30)
        assert not worker.is_alive(), "downloader worker hung"
    return dl


def feed_continuously(dl, urls, stop_event):
    """Keep topping the task queue up until stop_event is set.

    A pre-filled in-memory queue would drain in microseconds, letting idle
    workers take the empty-queue exit before ``max_num`` is ever reached. This
    feeder mimics a live parser that still has pages to emit, so the cap (not
    queue exhaustion) is what terminates the run.
    """
    it = iter(urls)
    while not stop_event.is_set():
        try:
            url = next(it)
        except StopIteration:
            return
        dl.in_queue.put({"file_url": url})
        time.sleep(0.005)


def run_capped(dl, urls, max_num, **kwargs):
    """Run until exactly max_num slots settle, fed by a live feeder thread."""
    dl.signal.set(parser_exited=False)
    stop = threading.Event()
    feeder = threading.Thread(target=feed_continuously, args=(dl, urls, stop), daemon=True)
    feeder.start()
    dl.start(
        file_idx_offset=kwargs.pop("file_idx_offset", 0),
        max_num=max_num,
        default_ext="jpg",
        queue_timeout=2,
        req_timeout=5,
        max_retry=3,
        overwrite=kwargs.pop("overwrite", False),
        max_idle_time=kwargs.pop("max_idle_time", None),
        **kwargs,
    )
    # Wait for the cap, then close the feeding side and clear surplus tasks so
    # workers blocked in queue.get() observe the stop signals and join.
    deadline = time.monotonic() + 60
    while not dl.signal.get("reach_max_num") and time.monotonic() < deadline:
        time.sleep(0.02)
    assert dl.signal.get("reach_max_num"), "cap was never reached"
    stop.set()
    dl.signal.set(parser_exited=True)
    while True:
        try:
            dl.in_queue.get_nowait()
            dl.in_queue.task_done()
        except queue_mod.Empty:
            break
    with dl._condition:
        dl._condition.notify_all()
    for worker in dl.workers:
        worker.join(timeout=30)
        assert not worker.is_alive(), "downloader worker hung"
    feeder.join(timeout=5)
    return dl


def files_of(tmp_path):
    return sorted(f for f in os.listdir(tmp_path) if not f.endswith(".tmp"))


def indices_of(tmp_path):
    return sorted(int(osp.splitext(f)[0]) for f in files_of(tmp_path))


def tmp_leftovers(tmp_path):
    return [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


# ---------------------------------------------------------------------------
# 1. 2xx happy path: the original inverted-status bug would fail every download
# ---------------------------------------------------------------------------
def test_two_xx_is_success(server, tmp_path):
    dl = build_downloader(tmp_path)
    enqueue(dl, [server.url("red.png"), server.url("green.jpg")])
    run(dl)

    assert dl.stats[STATUS_SUCCESS] == 2
    assert dl.fetched_num == 2
    # Indices are contiguous; extension assignment follows each URL regardless
    # of which worker wins the race.
    assert indices_of(tmp_path) == [0, 1]
    assert {n.split(".")[-1] for n in files_of(tmp_path)} == {"png", "jpg"}
    # Stored bytes are complete and decodable.
    for name in files_of(tmp_path):
        with Image.open(osp.join(tmp_path, name)) as img:
            img.load()
    assert dl.stats[STATUS_HTTP_ERROR] == 0
    assert dl.stats[STATUS_NETWORK_ERROR] == 0


def test_non_two_xx_is_http_error_and_uses_no_quota(server, tmp_path):
    dl = build_downloader(tmp_path, thread_num=1)
    enqueue(dl, [server.url("missing.png"), server.url("red.png")])
    run(dl)

    # 404 retried max_retry times (3), then counted once as http_error.
    assert dl.stats[STATUS_HTTP_ERROR] == 1
    assert server.attempts["missing.png"] == 3
    assert dl.stats[STATUS_SUCCESS] == 1
    # The failure burned neither a filename nor a quota slot.
    assert dl.fetched_num == 1
    assert files_of(tmp_path) == ["000000.png"]


# ---------------------------------------------------------------------------
# 2. Partial / failed writes never leave half files and never consume quota
# ---------------------------------------------------------------------------
def test_storage_failure_leaves_no_file_and_reuses_index(tmp_path):
    storage = FailingStorage(str(tmp_path), fail_times=1)
    session = FakeSession(
        {
            "http://x/a.png": [(GOOD_PNG, 200)],
            "http://x/b.png": [(GOOD_JPG, 200)],
        }
    )
    dl = build_downloader(tmp_path, thread_num=1, session=session, storage=storage)
    enqueue(dl, ["http://x/a.png", "http://x/b.png"])
    run(dl)

    assert storage.write_calls == 2
    assert dl.stats[STATUS_STORAGE_ERROR] == 1
    assert dl.stats[STATUS_SUCCESS] == 1
    assert dl.fetched_num == 1
    # The failed index was recycled: numbering starts at 000000, no gap.
    assert files_of(tmp_path) == ["000000.png"]
    assert tmp_leftovers(tmp_path) == []
    with Image.open(osp.join(tmp_path, "000000.png")) as img:
        img.load()
        assert img.size == (48, 48)


def test_atomic_write_failure_cleans_temp_and_keeps_target_intact(tmp_path, monkeypatch):
    """If the final rename fails, no half target and no temp file remain."""
    storage = FileSystem(str(tmp_path))
    # A healthy baseline file must stay byte-identical after the failed write.
    storage.write("000009.png", GOOD_JPG)
    baseline = open(osp.join(tmp_path, "000009.png"), "rb").read()

    real_replace = os.replace
    state = {"failures": 0}

    def failing_replace(src, dst):
        if osp.basename(dst) == "000000.png" and state["failures"] < 1:
            state["failures"] += 1
            raise OSError("simulated rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    session = FakeSession({"http://x/a.png": [(GOOD_PNG, 200)], "http://x/b.png": [(GOOD_JPG, 200)]})
    dl = build_downloader(tmp_path, thread_num=1, session=session, storage=storage)
    enqueue(dl, ["http://x/a.png", "http://x/b.png"])
    run(dl)

    assert dl.stats[STATUS_STORAGE_ERROR] == 1
    assert dl.stats[STATUS_SUCCESS] == 1
    assert tmp_leftovers(tmp_path) == []
    # The failed target was eventually created by the retry-recycled index,
    # and the pre-existing baseline file is byte-identical.
    assert osp.exists(osp.join(tmp_path, "000000.png"))
    assert open(osp.join(tmp_path, "000009.png"), "rb").read() == baseline
    assert indices_of(tmp_path) == [0, 9]


def test_enospc_sets_global_signal(tmp_path):
    enospc = OSError("no space left")
    enospc.errno = 28  # errno.ENOSPC
    storage = FailingStorage(str(tmp_path), fail_times=10, err=enospc)
    session = FakeSession({"http://x/a.png": [(GOOD_PNG, 200)]})
    dl = build_downloader(tmp_path, thread_num=1, session=session, storage=storage)
    enqueue(dl, ["http://x/a.png"])
    run(dl)

    assert dl.signal.get("exceed_storage_space") is True
    assert dl.stats[STATUS_STORAGE_ERROR] == 1
    assert dl.fetched_num == 0
    assert files_of(tmp_path) == []


def test_atomic_filesystem_write_no_tmp_residue(tmp_path):
    storage = FileSystem(str(tmp_path))
    storage.write("000000.png", GOOD_PNG)
    assert osp.isfile(osp.join(tmp_path, "000000.png"))
    assert tmp_leftovers(tmp_path) == []


# ---------------------------------------------------------------------------
# 3. Concurrency: max_num is an exact upper bound under many threads/tasks
# ---------------------------------------------------------------------------
def test_max_num_not_oversold_under_concurrency(server, tmp_path):
    dl = build_downloader(tmp_path, thread_num=8)
    # Distinct URLs (the task queue would de-duplicate identical tasks); a
    # live feeder keeps work available so the cap, not queue exhaustion, ends
    # the run. All aliases serve the same PNG payload.
    urls = [server.url(f"r{i}.png") for i in range(40)]
    run_capped(dl, urls, max_num=10)

    assert dl.fetched_num == 10
    assert dl.stats[STATUS_SUCCESS] == 10
    assert len(files_of(tmp_path)) == 10
    # Unique, contiguous 6-digit numbering.
    assert files_of(tmp_path) == [f"{i:06d}.png" for i in range(10)]
    assert dl.signal.get("reach_max_num") is True


def test_failures_among_many_tasks_still_have_gap_free_numbering(server, tmp_path):
    dl = build_downloader(tmp_path, thread_num=6)
    # Distinct URLs (the queue de-duplicates identical task dicts); aliases
    # share the underlying payloads.
    urls = (
        [server.url(f"b{i}.png") for i in range(3)]  # decode errors
        + [server.url(f"m{i}.png") for i in range(2)]  # 404s
        + [server.url(f"r{i}.png") for i in range(5)]  # good PNGs
        + [server.url(f"g{i}.jpg") for i in range(2)]  # good JPGs
    )
    enqueue(dl, urls)
    # Cap above the number of good payloads so every queued task is processed.
    run(dl, max_num=12)

    assert dl.stats[STATUS_SUCCESS] == 7
    assert dl.fetched_num == 7
    assert len(files_of(tmp_path)) == 7
    assert indices_of(tmp_path) == list(range(7))
    assert dl.stats[STATUS_DECODE_ERROR] == 3
    assert dl.stats[STATUS_HTTP_ERROR] == 2


# ---------------------------------------------------------------------------
# 4. Bad images: decode vs rejection are distinct and never hit the disk
# ---------------------------------------------------------------------------
def test_bad_and_truncated_images_are_decode_errors(tmp_path):
    session = FakeSession(
        {
            "http://x/bad.png": [(NOT_AN_IMAGE, 200)],
            "http://x/trunc.png": [(TRUNCATED_PNG, 200)],
            "http://x/good.png": [(GOOD_PNG, 200)],
        }
    )
    dl = build_downloader(tmp_path, thread_num=2, session=session)
    enqueue(dl, ["http://x/bad.png", "http://x/trunc.png", "http://x/good.png"])
    run(dl)

    assert dl.stats[STATUS_DECODE_ERROR] == 2
    assert dl.stats[STATUS_SUCCESS] == 1
    assert dl.fetched_num == 1
    assert files_of(tmp_path) == ["000000.png"]
    # Decode failures are not retried: exactly one attempt each.
    assert sum(1 for url, _ in session.calls if url.endswith("bad.png")) == 1
    assert sum(1 for url, _ in session.calls if url.endswith("trunc.png")) == 1


def test_size_filter_rejection_is_separate_from_decode(tmp_path):
    session = FakeSession(
        {
            "http://x/small.png": [(GOOD_PNG, 200)],  # 32x24
            "http://x/big.png": [(BIG_PNG, 200)],  # 200x200
        }
    )
    dl = build_downloader(tmp_path, thread_num=1, session=session)
    enqueue(dl, ["http://x/small.png", "http://x/big.png"])
    run(dl, min_size=(100, 100))

    assert dl.stats[STATUS_REJECTED] == 1
    assert dl.stats[STATUS_DECODE_ERROR] == 0
    assert dl.stats[STATUS_SUCCESS] == 1
    assert files_of(tmp_path) == ["000000.png"]
    with Image.open(osp.join(tmp_path, "000000.png")) as img:
        assert img.size == (200, 200)


# ---------------------------------------------------------------------------
# 5. Retries: network exceptions and HTTP errors retry, then count distinctly
# ---------------------------------------------------------------------------
def test_network_exception_retried_then_recorded(tmp_path):
    session = FakeSession(
        {
            "http://x/flaky.png": [
                requests.ConnectionError("boom"),
                requests.Timeout("slow"),
                (GOOD_PNG, 200),
            ]
        }
    )
    dl = build_downloader(tmp_path, thread_num=1, session=session)
    enqueue(dl, ["http://x/flaky.png"])
    run(dl)

    assert len(session.calls) == 3
    assert dl.stats[STATUS_SUCCESS] == 1
    assert dl.stats[STATUS_NETWORK_ERROR] == 0
    assert dl.fetched_num == 1


def test_network_exception_exhausting_retries_uses_no_quota(tmp_path):
    session = FakeSession(
        {"http://x/dead.png": [requests.ConnectionError("boom")] * 3, "http://x/good.png": [(GOOD_PNG, 200)]}
    )
    dl = build_downloader(tmp_path, thread_num=1, session=session)
    enqueue(dl, ["http://x/dead.png", "http://x/good.png"])
    run(dl)

    assert len([c for c in session.calls if c[0].endswith("dead.png")]) == 3
    assert dl.stats[STATUS_NETWORK_ERROR] == 1
    assert dl.stats[STATUS_SUCCESS] == 1
    assert dl.fetched_num == 1
    assert files_of(tmp_path) == ["000000.png"]


def test_http_flaky_endpoint_succeeds_after_retries(server, tmp_path):
    dl = build_downloader(tmp_path, thread_num=1)
    enqueue(dl, [server.url("flaky.png")])
    run(dl)

    assert server.attempts["flaky.png"] == 3  # 2 failures + 1 success
    assert dl.stats[STATUS_HTTP_ERROR] == 0
    assert dl.stats[STATUS_SUCCESS] == 1
    assert files_of(tmp_path) == ["000000.png"]


# ---------------------------------------------------------------------------
# 6. Reruns: overwrite / skip / file_idx_offset=auto semantics
# ---------------------------------------------------------------------------
def test_rerun_skips_existing_files_without_overwrite(server, tmp_path):
    # A single worker makes the URL -> index mapping deterministic across
    # runs, which is the ordered re-crawl the skip semantics are designed for.
    dl1 = build_downloader(tmp_path, thread_num=1)
    enqueue(dl1, [server.url("red.png"), server.url("green.jpg")])
    run(dl1)
    first_files = files_of(tmp_path)
    assert len(first_files) == 2

    # Second run over the same URLs and directory, overwrite stays False.
    # Each task's leading indices are occupied, so they are skipped (and
    # counted against the quota) before the payload is written at the next
    # free index; nothing existing is overwritten.
    dl2 = build_downloader(tmp_path, thread_num=1)
    enqueue(dl2, [server.url("red.png"), server.url("green.jpg")])
    run(dl2)

    assert dl2.stats[STATUS_SKIPPED] == 2
    assert dl2.stats[STATUS_SUCCESS] == 2
    # Original files are byte-identical; only new higher-index files appear.
    for name in first_files:
        assert osp.exists(osp.join(tmp_path, name))
    assert indices_of(tmp_path) == [0, 1, 2, 3]
    # A skipped slot settles against the quota exactly like a download.
    assert dl2.fetched_num == 4


def test_overwrite_redownloads_and_replaces(server, tmp_path):
    url = server.url("red.png")
    dl1 = build_downloader(tmp_path, thread_num=1)
    enqueue(dl1, [url])
    run(dl1)
    target = osp.join(tmp_path, "000000.png")
    first_bytes = open(target, "rb").read()
    assert first_bytes == GOOD_PNG

    # overwrite=True re-fetches the same URL and atomically replaces the file.
    dl2 = build_downloader(tmp_path, thread_num=1)
    enqueue(dl2, [url])
    run(dl2, overwrite=True)

    assert dl2.stats[STATUS_SUCCESS] == 1
    assert dl2.stats[STATUS_SKIPPED] == 0
    assert files_of(tmp_path) == ["000000.png"]
    assert open(target, "rb").read() == first_bytes
    with Image.open(target) as img:
        assert img.size == (32, 24)


def test_file_idx_offset_auto_continues_after_existing_files(server, tmp_path):
    # Seed the directory with three .png files, like a previous run that
    # consumed .png URLs; max index is 2.
    for i, payload in enumerate([GOOD_PNG, GOOD_JPG, BIG_PNG]):
        FileSystem(str(tmp_path)).write(f"{i:06d}.png", payload)

    dl = build_downloader(tmp_path, thread_num=2)
    enqueue(dl, [server.url("red.png"), server.url("blue.png")])
    run(dl, file_idx_offset="auto")

    assert dl.file_idx_offset == 2  # max existing index
    assert dl.stats[STATUS_SUCCESS] == 2
    # auto continues numbering at offset+1: new files are 000003, 000004.
    assert indices_of(tmp_path) == [0, 1, 2, 3, 4]


def test_integer_file_idx_offset_shifts_names(tmp_path):
    session = FakeSession({"http://x/a.png": [(GOOD_PNG, 200)]})
    dl = build_downloader(tmp_path, thread_num=1, session=session)
    enqueue(dl, ["http://x/a.png"])
    run(dl, file_idx_offset=10)
    assert files_of(tmp_path) == ["000010.png"]


def test_rerun_with_failures_then_success_keeps_contiguous_numbering(server, tmp_path):
    dl = build_downloader(tmp_path, thread_num=1)
    enqueue(
        dl,
        [server.url("bad.png"), server.url("missing.png"), server.url("red.png"), server.url("green.jpg")],
    )
    run(dl)
    assert dl.fetched_num == 2
    assert indices_of(tmp_path) == [0, 1]

    # Rerun: the previously failing payload still fails (no quota consumed),
    # while the fresh good payload first walks over (skips) the two indices
    # occupied by run one and then takes the next contiguous index.
    dl2 = build_downloader(tmp_path, thread_num=1)
    enqueue(dl2, [server.url("bad.png"), server.url("blue.png")])
    run(dl2)
    assert dl2.stats[STATUS_DECODE_ERROR] == 1
    assert dl2.stats[STATUS_SKIPPED] == 2  # indices 0 and 1 already occupied
    assert dl2.stats[STATUS_SUCCESS] == 1
    assert dl2.fetched_num == 3
    assert indices_of(tmp_path) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 7. process_meta only fires for committed files; stats reconcile on exit
# ---------------------------------------------------------------------------
def test_process_meta_called_only_for_success(tmp_path):
    session = FakeSession(
        {
            "http://x/bad.png": [(NOT_AN_IMAGE, 200)],
            "http://x/good.png": [(GOOD_PNG, 200)],
        }
    )

    class MetaDownloader(ImageDownloader):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.meta_seen = []

        def process_meta(self, task):
            self.meta_seen.append((task["filename"], task["status"]))

    dl = MetaDownloader(
        thread_num=2,
        signal=Signal(),
        session=session,
        storage=FileSystem(str(tmp_path)),
    )
    dl.signal.set(feeder_exited=False, parser_exited=False, reach_max_num=False, exceed_storage_space=False)
    enqueue(dl, ["http://x/bad.png", "http://x/good.png"])
    run(dl)

    assert dl.meta_seen == [("000000.png", STATUS_SUCCESS)]


def test_stats_reconcile_on_thread_exit(server, tmp_path):
    dl = build_downloader(tmp_path, thread_num=5)
    # Distinct URLs to avoid queue de-duplication; 7 successes, 3 decode
    # failures and 1 HTTP failure, all fully drained before shutdown.
    urls = (
        [server.url(f"r{i}.png") for i in range(7)]
        + [server.url(f"b{i}.png") for i in range(3)]
        + [server.url("missing.png")]
    )
    enqueue(dl, urls)
    # Cap above the total task count so every task reaches a final outcome.
    run(dl, max_num=20)

    # Every queued task is accounted for exactly once across the stats.
    assert sum(dl.stats.values()) == len(urls)
    assert dl.stats[STATUS_SUCCESS] == 7
    assert dl.stats[STATUS_DECODE_ERROR] == 3
    assert dl.stats[STATUS_HTTP_ERROR] == 1
    # Slot-consuming outcomes match the settled-file counter exactly.
    assert dl.stats[STATUS_SUCCESS] + dl.stats[STATUS_SKIPPED] == dl.fetched_num == 7
    assert all(not w.is_alive() for w in dl.workers)


def test_max_idle_time_does_not_fire_while_downloads_succeed(server, tmp_path):
    dl = build_downloader(tmp_path, thread_num=2)
    enqueue(dl, [server.url("red.png"), server.url("green.jpg"), server.url("blue.png")])
    run(dl, max_idle_time=1)
    assert dl.stats[STATUS_SUCCESS] == 3
    assert dl.fetched_num == 3


# ---------------------------------------------------------------------------
# 8. End-to-end through the real Crawler (Feeder -> PseudoParser -> Downloader)
#    using the production Session and FileSystem storage.
# ---------------------------------------------------------------------------
def test_end_to_end_crawler_cap_decode_and_rerun(server, tmp_path):
    from icrawler.builtin import UrlListCrawler

    urls = [
        server.url("red.png"),
        server.url("green.jpg"),
        server.url("bad.png"),  # undecodable
        server.url("blue.png"),
        server.url("red.png"),
    ]
    crawler = UrlListCrawler(
        downloader_threads=3,
        storage={"backend": "FileSystem", "root_dir": str(tmp_path)},
        log_level=40,
    )
    crawler.crawl(urls, max_num=3)
    dl = crawler.downloader

    assert dl.fetched_num == 3  # cap never exceeded despite 5 tasks
    assert dl.stats[STATUS_SUCCESS] == 3
    assert dl.stats[STATUS_DECODE_ERROR] == 1
    assert indices_of(tmp_path) == [0, 1, 2]
    assert tmp_leftovers(tmp_path) == []

    # Rerun into the same directory without overwrite: occupied indices are
    # skipped and a fresh good download takes the next contiguous index.
    crawler2 = UrlListCrawler(
        downloader_threads=1,
        storage={"backend": "FileSystem", "root_dir": str(tmp_path)},
        log_level=40,
    )
    crawler2.crawl([server.url("red.png")], max_num=5)
    assert crawler2.downloader.stats[STATUS_SKIPPED] == 3
    assert crawler2.downloader.stats[STATUS_SUCCESS] == 1
    assert indices_of(tmp_path) == [0, 1, 2, 3]
    assert tmp_leftovers(tmp_path) == []


def test_end_to_end_crawler_retry_then_success(server, tmp_path):
    from icrawler.builtin import UrlListCrawler

    crawler = UrlListCrawler(
        downloader_threads=1,
        storage={"backend": "FileSystem", "root_dir": str(tmp_path)},
        log_level=40,
    )
    # flaky.png fails twice with 500 then serves a valid image.
    crawler.crawl([server.url("flaky.png")], max_num=5)
    dl = crawler.downloader
    assert dl.stats[STATUS_SUCCESS] == 1
    assert dl.stats[STATUS_HTTP_ERROR] == 0
    assert server.attempts["flaky.png"] == 3
    assert dl.fetched_num == 1
    assert indices_of(tmp_path) == [0]
