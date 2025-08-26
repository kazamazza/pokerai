from typing import Optional, Union
import torch

DeviceLike = Union[str, torch.device]

def to_device(device: Optional[DeviceLike]) -> torch.device:
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")
    return torch.device(device)