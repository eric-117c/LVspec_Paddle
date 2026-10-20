"""Keep host tests independent of Paddle's GPU-build default device."""

import paddle
import pytest


@pytest.fixture(autouse=True)
def default_cpu_device():
    previous_device, previous_dtype = paddle.get_device(), paddle.get_default_dtype()
    paddle.set_device("cpu")
    paddle.set_default_dtype("float32")
    try:
        yield
    finally:
        paddle.set_default_dtype(previous_dtype)
        paddle.set_device(previous_device)
