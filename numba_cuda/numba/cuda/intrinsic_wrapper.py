from .decorators import jit
import numba


@jit(device=True)
def all_sync(mask, predicate):
    """
    If for all threads in the masked warp the predicate is true, then
    a non-zero value is returned, otherwise 0 is returned.
    """
    return numba.cuda.vote_sync_intrinsic(mask, 0, predicate)[1]


@jit(device=True)
def any_sync(mask, predicate):
    """
    If for any thread in the masked warp the predicate is true, then
    a non-zero value is returned, otherwise 0 is returned.
    """
    return numba.cuda.vote_sync_intrinsic(mask, 1, predicate)[1]


@jit(device=True)
def eq_sync(mask, predicate):
    """
    If for all threads in the masked warp the boolean predicate is the same,
    then a non-zero value is returned, otherwise 0 is returned.
    """
    return numba.cuda.vote_sync_intrinsic(mask, 2, predicate)[1]


@jit(device=True)
def ballot_sync(mask, predicate):
    """
    Returns a mask of all threads in the warp whose predicate is true,
    and are within the given mask.
    """
    return numba.cuda.vote_sync_intrinsic(mask, 3, predicate)[0]
