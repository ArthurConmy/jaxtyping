import pickle

import cloudpickle
import numpy as np
import pytest


try:
    import torch
except ImportError:
    torch = None

# Import typecheckers conditionally
try:
    import beartype
except ImportError:
    beartype = None

try:
    import typeguard
except ImportError:
    typeguard = None


from jaxtyping import AbstractArray, Array, Float, Shaped, jaxtyped


def test_pickle():
    for p in (pickle, cloudpickle):
        x = p.dumps(Shaped[Array, ""])
        y = p.loads(x)
        assert y.dtype is Shaped
        assert y.dim_str == ""

        x = p.dumps(AbstractArray)
        y = p.loads(x)
        assert y is AbstractArray

        x = p.dumps(Shaped[np.ndarray, "3 4 hi"])
        y = p.loads(x)
        assert y.dtype is Shaped
        assert y.dim_str == "3 4 hi"

        if torch is not None:
            x = p.dumps(Float[torch.Tensor, "batch length"])
            y = p.loads(x)
            assert y.dtype is Float
            assert y.dim_str == "batch length"


# Helper to get the actual decorator function from the library module
def get_typechecker(typechecker_lib):
    if typechecker_lib is None:
        return None
    if typechecker_lib is typeguard:
        # Handle different versions/APIs of typeguard
        if hasattr(typeguard, "typechecked"):
            return typeguard.typechecked
        return None
    elif typechecker_lib is beartype:
        return beartype.beartype
    return None

# Parameterize the test over available typecheckers
@pytest.mark.parametrize("typechecker_lib", [beartype, typeguard])
def test_jaxtyped_cloudpickle(typechecker_lib):
    # MWE derived from https://github.com/patrick-kidger/jaxtyping/issues/332
    # and https://github.com/patrick-kidger/jaxtyping/issues/343

    typechecker = get_typechecker(typechecker_lib)
    
    assert typechecker is not None

    # Define a function locally to ensure cloudpickle handles the closure.
    @jaxtyped(typechecker=typechecker)
    def local_typechecked_fn(x: Float[np.ndarray, "d"]) -> float:
        # Use np.sum() to ensure numpy interaction works
        return float(np.sum(x))

    # 1. Test functionality locally
    arr = np.array([1.0, 2.0], dtype=np.float64) 
    assert local_typechecked_fn(arr) == 3.0

    # 2. Test serialization
    try:
        dumped = cloudpickle.dumps(local_typechecked_fn)
    except Exception as e:
        pytest.fail(f"Cloudpickle failed to serialize jaxtyped function with {typechecker_lib.__name__}: {e}")

    # 3. Test deserialization and functionality
    fn = cloudpickle.loads(dumped)
    assert fn(arr) == 3.0

    # 4. Ensure type checking still works after deserialization
    arr_bad_shape = np.array([[1.0]], dtype=np.float64)
    # We expect a TypeCheckError (from jaxtyping) or a specific error from the checker.
    # Catching Exception as the specific type varies (e.g. beartype.roar.BeartypeCallHintParamViolation).
    with pytest.raises(Exception):
        fn(arr_bad_shape)
    
    # 5. Verify the weakref mechanism (used for __no_type_check__) works after deserialization
    assert getattr(fn, "__no_type_check__", False) is False
    fn.__no_type_check__ = True
    # Should now execute without error even with bad shape
    assert fn(arr_bad_shape) == 1.0

def test_jaxtyped_cloudpickle_no_typechecker():
    # Test the case where typechecker=None. This path does not use the weakref mechanism
    # but we ensure it remains picklable.
    @jaxtyped(typechecker=None)
    def f(x):
      return x + 1

    assert f(5) == 6
    dumped = cloudpickle.dumps(f)
    loaded_f = cloudpickle.loads(dumped)
    assert loaded_f(10) == 11
