""" CUDA / AMP utils
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch


def resolve_inference_device(device_name: str):
    """
    Map config device string to torch.device. If CUDA is requested but no GPU
    is available, fall back to CPU (common on laptops / CPU-only installs).
    """
    if (
        isinstance(device_name, str)
        and device_name.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        return "cpu", torch.device("cpu")
    return device_name, torch.device(device_name)


def get_devices_info(device_names="0"):

    if device_names.startswith('cuda'):
        device_names = device_names.split('cuda:')[1]
    elif device_names.startswith('cpu'):
        return "CPU"

    devices_info = ""
    for i, device_id in enumerate(device_names.split(',')):
        p = torch.cuda.get_device_properties(i)
        # bytes to MB
        devices_info += f"CUDA:{device_id} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"
    return devices_info
