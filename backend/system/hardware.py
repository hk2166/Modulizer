import platform
import shutil

import psutil
import torch


def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}

    props = torch.cuda.get_device_properties(0)
    total_vram = round(props.total_memory / (1024 ** 3), 2)

    return {
        "cuda": True,
        "gpu_name": props.name,
        "vram_gb": total_vram,
        "low_vram_mode": total_vram <= 6,
    }


def get_ram_info() -> dict:
    ram = psutil.virtual_memory()
    return {
        "total_ram_gb": round(ram.total / (1024 ** 3), 2),
        "available_ram_gb": round(ram.available / (1024 ** 3), 2),
    }


def get_disk_info() -> dict:
    disk = shutil.disk_usage("/")
    return {
        "free_disk_gb": round(disk.free / (1024 ** 3), 2),
    }


def get_system_info() -> dict:
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
    }


def get_full_system_profile() -> dict:
    return {
        "system": get_system_info(),
        "gpu": get_gpu_info(),
        "ram": get_ram_info(),
        "disk": get_disk_info(),
    }
