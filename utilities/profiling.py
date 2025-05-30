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
