"""Native Paddle inference for LVSpec (no PyTorch model dependency)."""

__all__ = ["load_model"]


def load_model(*args, **kwargs):
    from .checkpoint import load_model as load

    return load(*args, **kwargs)
