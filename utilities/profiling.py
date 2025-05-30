from contextlib import nullcontext
import torch.profiler as tpf


def optional_profiler(is_profiling: bool, device: str):
    if is_profiling:
        profile_activities = [tpf.ProfilerActivity.CPU]
        if device == "cuda":
            profile_activities.append(tpf.ProfilerActivity.CUDA)
        return tpf.profile(activities=profile_activities, record_shapes=False)
    else:
        return nullcontext()


def get_total_time(events):
    # Snippet taken from torch.profiler source code (with modifications)
    sum_self_cpu_time_total = 0
    sum_self_device_time_total = 0
    for evt in events:
        sum_self_cpu_time_total += evt.self_cpu_time_total
        if str(evt.device_type) == "DeviceType.CPU" and evt.is_legacy:
            # in legacy profiler, kernel info is stored in cpu events
            sum_self_device_time_total += evt.self_device_time_total
        elif str(evt.device_type) == "DeviceType.CUDA" and not evt.is_user_annotation:
            # in kineto profiler, there're events with the correct device type (e.g. CUDA)
            sum_self_device_time_total += evt.self_device_time_total
    return sum_self_cpu_time_total / 1000, sum_self_device_time_total / 1000
