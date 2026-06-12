"""Run heavy pure-Python EV work in persistent worker subprocesses.

The exact-EV engine is pure Python; on a GIL build a pre-deal sweep
(~15 s of solid bytecode) on a background *thread* steals the interpreter
from the Tk event loop and the GUI visibly stutters. Each call here ships
the computation to a small persistent child process instead (args are
tuples/frozen dataclasses, results floats/dicts — pickling is trivial), so
the parent thread just blocks on a future and releases the GIL.

Two pools: "advice" (per-seat advise + warm-ups, latency-sensitive,
single worker) and "predeal" (the sweeps, constants.PREDEAL_WORKERS
workers — the sweep fans out one job per dealer up-card via run_many,
V3 E2) — a sweep must never queue a seat's advice behind it. Workers
spawn lazily on first use and are reused; ev_engine's thread-local memos
accumulate across calls inside each child (single-worker pools run every
job on the same child thread; the predeal workers each keep their own
caches, warmed by whichever up-card jobs land on them).

On any pool failure (spawn blocked, child killed, interpreter shutdown)
the call falls back to computing in-process — GIL-noisy but always
correct. Set BJ_EV_INPROC=1 to disable subprocesses entirely (debugging,
constrained environments).
"""

import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pickle import PicklingError

from ..common import constants

_pools = {}
_lock = threading.Lock()


def _enabled() -> bool:
    return os.environ.get("BJ_EV_INPROC", "") != "1"


def _workers(name) -> int:
    """Worker count for a named pool — read at pool creation time."""
    if name == "predeal":
        return max(1, int(getattr(constants, "PREDEAL_WORKERS", 1)))
    return 1


def _pool(name):
    with _lock:
        pool = _pools.get(name)
        if pool is None:
            pool = ProcessPoolExecutor(max_workers=_workers(name))
            _pools[name] = pool
        return pool


def _discard(name):
    with _lock:
        pool = _pools.pop(name, None)
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def run(name, fn, *args):
    """fn(*args) in the named worker process; in-process on any pool failure.

    fn must be a module-level function and args picklable. Exceptions raised
    by fn itself propagate to the caller either way."""
    if not _enabled():
        return fn(*args)
    try:
        future = _pool(name).submit(fn, *args)
    except BrokenProcessPool as e:
        # An idle child died (AV kill, crash): submit() itself raises once
        # the pool is flagged broken — and BrokenProcessPool subclasses
        # RuntimeError, so it must be discarded HERE or the broken executor
        # stays cached and every later call falls back in-process forever.
        _discard(name)
        print(f"EV offload worker '{name}' died ({e}); computing in-process.")
        return fn(*args)
    except (RuntimeError, OSError) as e:  # interpreter shutdown / spawn fail
        print(f"EV offload pool '{name}' unavailable ({e}); computing in-process.")
        return fn(*args)
    try:
        return future.result()
    except BrokenProcessPool as e:  # child died; rebuild lazily on next call
        _discard(name)
        print(f"EV offload worker '{name}' died ({e}); computing in-process.")
        return fn(*args)
    except (PicklingError, AttributeError, TypeError):
        # fn, an argument, or the result can't cross the process boundary
        # (test mocks, closures) — compute in-process instead. A TypeError
        # raised by fn itself just gets re-raised from the in-process run.
        return fn(*args)


def run_many(name, fn, args_list):
    """[fn(*args) for args in args_list], fanned across the named pool's
    workers concurrently; falls back to sequential in-process computation
    on any pool failure (same contract as run()). fn must be module-level
    and every args tuple picklable."""
    args_list = list(args_list)
    if not _enabled() or not args_list:
        return [fn(*args) for args in args_list]
    try:
        futures = [_pool(name).submit(fn, *args) for args in args_list]
    except BrokenProcessPool as e:
        _discard(name)
        print(f"EV offload worker '{name}' died ({e}); computing in-process.")
        return [fn(*args) for args in args_list]
    except (RuntimeError, OSError) as e:
        print(f"EV offload pool '{name}' unavailable ({e}); computing in-process.")
        return [fn(*args) for args in args_list]
    try:
        return [future.result() for future in futures]
    except BrokenProcessPool as e:
        _discard(name)
        print(f"EV offload worker '{name}' died ({e}); computing in-process.")
        return [fn(*args) for args in args_list]
    except (PicklingError, AttributeError, TypeError):
        # Something couldn't cross the process boundary (test mocks,
        # closures) — compute everything in-process instead.
        return [fn(*args) for args in args_list]


def prewarm(*names):
    """Spawn the named workers now (call off the Tk thread) so the first
    real EV job doesn't pay the process start-up."""
    if not _enabled():
        return
    for name in names:
        try:
            _pool(name).submit(int)
        except (RuntimeError, OSError):
            pass


def shutdown():
    """Stop all workers (tests / app exit; safe to call twice)."""
    with _lock:
        pools, _pools_snapshot = dict(_pools), _pools.clear()
    for pool in pools.values():
        pool.shutdown(wait=False, cancel_futures=True)
