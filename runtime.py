import torch


def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    if device_arg == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    if device_arg == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {device_arg}")


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}

    allocated_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {
        "peak_cuda_allocated_mib": allocated_mib,
        "peak_cuda_reserved_mib": reserved_mib,
    }
