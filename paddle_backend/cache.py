"""Bounded, per-layer KV storage with speculative rollback and compaction."""

import paddle

from .ops import device_of, on_device


class KVCache:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("KV capacity must be positive")
        self.capacity = capacity
        self.length = 0
        self.key = self.value = None

    def append(self, key, value):
        count = key.shape[2]
        end = self.length + count
        if key.shape != value.shape or end > self.capacity:
            raise ValueError(f"Invalid KV append: length={self.length}, count={count}, capacity={self.capacity}")
        if self.key is None:
            shape = list(key.shape)
            shape[2] = self.capacity
            with paddle.device.device_guard(device_of(key)):
                self.key = paddle.empty(shape, dtype=key.dtype)
                self.value = paddle.empty(shape, dtype=value.dtype)
        # Slice assignment writes the original allocation; no concatenation of the prefix.
        self.key[:, :, self.length:end, :] = key
        self.value[:, :, self.length:end, :] = value
        self.length = end
        return self.key[:, :, :end, :], self.value[:, :, :end, :]

    def truncate(self, length):
        if not 0 <= length <= self.length:
            raise ValueError(f"Cannot truncate KV length {self.length} to {length}")
        self.length = int(length)

    @on_device("key")
    def compact(self, indices, prefix_length):
        indices = [int(i) for i in indices]
        if not 0 <= prefix_length <= self.length:
            raise ValueError("Invalid cache prefix")
        if any(i < prefix_length or i >= self.length for i in indices):
            raise ValueError("Accepted cache indices must belong to the speculative suffix")
        if prefix_length + len(indices) > self.capacity:
            raise ValueError("Accepted cache suffix exceeds capacity")
        if indices:
            idx = paddle.to_tensor(indices, dtype="int64", place=self.key.place)
            # Gather both sources before overwriting potentially overlapping destinations.
            key = paddle.index_select(self.key, idx, axis=2)
            value = paddle.index_select(self.value, idx, axis=2)
            end = prefix_length + len(indices)
            self.key[:, :, prefix_length:end, :] = key
            self.value[:, :, prefix_length:end, :] = value
        self.length = prefix_length + len(indices)


class ModelCache:
    def __init__(self, num_layers, capacity):
        self.layers = [KVCache(capacity) for _ in range(num_layers)]

    @property
    def length(self):
        lengths = {layer.length for layer in self.layers}
        if len(lengths) != 1:
            raise RuntimeError(f"Inconsistent per-layer KV lengths: {lengths}")
        return next(iter(lengths))

    def truncate(self, length):
        for layer in self.layers:
            layer.truncate(length)

    def compact(self, indices, prefix_length):
        for layer in self.layers:
            layer.compact(indices, prefix_length)
