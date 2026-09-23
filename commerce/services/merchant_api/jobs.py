"""One seller-local executor, shared by FL and personalization callers."""
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable, TypeVar

from commerce.packages.contracts.errors import JobBusyError

T = TypeVar("T")


class SellerJobs:
    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="seller-job")
        self._busy = Lock()

    def submit(self, operation: Callable[[], T]) -> Future[T]:
        if not self._busy.acquire(blocking=False):
            raise JobBusyError("A seller training task is already running")
        def run():
            try:
                return operation()
            finally:
                self._busy.release()
        try:
            future = self._pool.submit(run)
        except BaseException:
            self._busy.release()
            raise
        def on_cancel(done):
            if done.cancelled():
                self._busy.release()
        future.add_done_callback(on_cancel)
        return future

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
