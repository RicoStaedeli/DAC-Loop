import torch


class DeviceResolver:
    @staticmethod
    def get() -> torch.device:
        """
        Resolves and returns the appropriate torch device.
        Prioritizes MPS, then CUDA, and defaults to CPU if neither is available.
        Returns:
            torch.device: The resolved device.
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
