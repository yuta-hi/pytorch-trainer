import numbers
import typing as tp  # NOQA
import typing_extensions as tpe  # NOQA

try:
    from typing import TYPE_CHECKING  # NOQA
except ImportError:
    # typing.TYPE_CHECKING doesn't exist before Python 3.5.2
    TYPE_CHECKING = False


Shape = tp.Tuple[int, ...]


ShapeSpec = tp.Union[int, tp.Sequence[int]]  # Sequence includes Tuple[int, ...] # NOQA


DTypeSpec = tp.Union[tp.Any]  # TODO(okapies): encode numpy.dtype


NdArray = tp.Union[
    'numpy.ndarray',
]
"""The ndarray types supported in :func:`pytorch_trainer.get_array_types`
"""


Xp = tp.Union[tp.Any]  # TODO(okapies): encode numpy/cupy/ideep/chainerx


class AbstractInitializer(tpe.Protocol):
    """Protocol class for Initializer.

    It can be either an :class:`pytorch_trainer.Initializer` or a callable object
    that takes an ndarray.

    This is only for PEP 544 compliant static type checkers.
    """
    dtype = None  # type: tp.Optional[DTypeSpec]

    def __call__(self, array: NdArray) -> None:
        pass


ScalarValue = tp.Union[
    'numpy.generic',
    bytes,
    str,
    memoryview,
    numbers.Number,
]
"""The scalar types supported in :func:`numpy.isscalar`.
"""


InitializerSpec = tp.Union[AbstractInitializer, ScalarValue, 'numpy.ndarray']


DeviceSpec = tp.Union[
    'backend.Device',
    str,
    tp.Tuple[str, int],
    tp.Tuple['ModuleType', int],  # cupy module and device ID
]
"""The device specifier types supported in :func:`pytorch_trainer.get_device`
"""
# TODO(okapies): Use Xp instead of ModuleType


CudaDeviceSpec = tp.Union['cuda.Device', int, 'numpy.integer']  # NOQA
"""
This type only for the deprecated :func:`pytorch_trainer.cuda.get_device` API.
Use :class:`~pytorch_trainer.types.DeviceSpec` instead.
"""
