"""Host-to-device copies that do not make the host wait.

A blocking copy waits until the device has run everything queued before it. On
a GPU that several runs share, that wait is this run's next turn, about 9 ms on
the lab server even for a few bytes, and a training step made six of them. A
copy from pinned memory with non_blocking=True is queued like a kernel instead.
PyTorch's pinned-memory cache keeps the source until the copy has run.
"""
import torch


def to_device(tensor: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    if device.type != "cuda" or tensor.device.type != "cpu":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)
