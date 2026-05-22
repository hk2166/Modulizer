import platform
import shutil
import subprocess
import psutil
import torch


# GPU power lookup. TDP / power cap in watts, keyed on a substring of the
# GPU name. Used when nvidia-smi can't be reached. Order matters — more
# specific keys first so e.g. "RTX 4090 Laptop" wins over "RTX 4090".
_GPU_POWER_FALLBACK = {
    # Data center / AI accelerators
    "B200":             1000,
    "B100":              700,
    "GB200":            1200,
    "H200 NVL":          600,
    "H200":              700,
    "H100 NVL":          400,
    "H100":              700,
    "L40S":              350,
    "L40":               300,
    "L4":                 72,
    "A100 80GB":         400,
    "A100":              400,
    "A40":               300,
    "A30":               165,
    "A16":               250,
    "A10":               150,
    "A2":                 60,
    "V100":              300,
    "T4":                 70,
    "P100":              250,
    "P40":               250,
    "P4":                 75,
    # Workstation / Pro
    "RTX 6000 Ada":      300,
    "RTX 5000 Ada":      250,
    "RTX 4500 Ada":      210,
    "RTX 4000 Ada":      130,
    "RTX A6000":         300,
    "RTX A5500":         230,
    "RTX A5000":         230,
    "RTX A4500":         200,
    "RTX A4000":         140,
    "RTX A2000":          70,
    # Consumer laptop
    "RTX 5090 Laptop":   175,
    "RTX 5080 Laptop":   175,
    "RTX 5070 Ti Laptop":140,
    "RTX 5070 Laptop":   115,
    "RTX 4090 Laptop":   175,
    "RTX 4080 Laptop":   150,
    "RTX 4070 Laptop":   115,
    "RTX 4060 Laptop":   115,
    "RTX 4050 Laptop":   100,
    "RTX 3080 Ti Laptop":175,
    "RTX 3080 Laptop":   165,
    "RTX 3070 Ti Laptop":150,
    "RTX 3070 Laptop":   140,
    "RTX 3060 Laptop":   115,
    "RTX 3050 Ti Laptop":100,
    "RTX 3050 Laptop":    90,
    "RTX 2080 Laptop":   150,
    "RTX 2070 Laptop":   115,
    "RTX 2060 Laptop":    90,
    "GTX 1660 Ti Laptop": 80,
    "GTX 1660 Laptop":    80,
    "GTX 1650 Ti":        55,
    "GTX 1650":           50,
    # Consumer desktop
    "RTX 5090":          575,
    "RTX 5080":          360,
    "RTX 5070 Ti":       300,
    "RTX 5070":          250,
    "RTX 4090":          450,
    "RTX 4080 SUPER":    320,
    "RTX 4080":          320,
    "RTX 4070 Ti SUPER": 285,
    "RTX 4070 Ti":       285,
    "RTX 4070 SUPER":    220,
    "RTX 4070":          200,
    "RTX 4060 Ti":       160,
    "RTX 4060":          115,
    "RTX 3090 Ti":       450,
    "RTX 3090":          350,
    "RTX 3080 Ti":       350,
    "RTX 3080":          320,
    "RTX 3070 Ti":       290,
    "RTX 3070":          220,
    "RTX 3060 Ti":       200,
    "RTX 3060":          170,
    "RTX 3050":          130,
    "RTX 2080 Ti":       250,
    "RTX 2080 SUPER":    250,
    "RTX 2080":          215,
    "RTX 2070 SUPER":    215,
    "RTX 2070":          175,
    "RTX 2060 SUPER":    175,
    "RTX 2060":          160,
    "GTX 1660 Ti":       120,
    "GTX 1660 SUPER":    125,
    "GTX 1660":          120,
    "GTX 1080 Ti":       250,
    "GTX 1080":          180,
    "GTX 1070 Ti":       180,
    "GTX 1070":          150,
    "GTX 1060":          120,
    "GTX 1050 Ti":        75,
    "GTX 1050":           75,
}


def _get_power_cap_watts(gpu_name: str | None) -> int | None:
    """
    GPU power cap in watts. Tries nvidia-smi first (works even when CUDA
    init is wedged), falls back to the static lookup, returns None if
    neither source has an answer.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=power.max_limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        )
        if out.returncode == 0 and out.stdout.strip():
            first = out.stdout.strip().splitlines()[0].strip()
            if first and not first.startswith("["):
                return int(round(float(first)))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    if gpu_name:
        for key, watts in _GPU_POWER_FALLBACK.items():
            if key.lower() in gpu_name.lower():
                return watts

    return None


def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        # nvidia-smi often still works when CUDA init is broken (e.g.
        # suspend/resume on Optimus laptops). Pass None so we skip the
        # name-based lookup and only trust the live read.
        return {
            "cuda": False,
            "power_watts": _get_power_cap_watts(None),
        }

    props = torch.cuda.get_device_properties(0)
    total_vram = round(props.total_memory / (1024 ** 3), 2)

    return {
        "cuda": True,
        "gpu_name": props.name,
        "vram_gb": total_vram,
        "low_vram_mode": total_vram <= 6,
        "power_watts": _get_power_cap_watts(props.name),
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
