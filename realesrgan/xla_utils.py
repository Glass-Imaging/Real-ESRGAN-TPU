import os
import torch.distributed as dist
import torch.multiprocessing as mp


if os.environ.get("PJRT_DEVICE"):
    from torch_xla.core import xla_model
    import torch_xla.experimental.pjrt_backend  # Needed for init_process_group even though otherwise unused.


def init_xla_dist():
    # Using PJRT dist so that calls like dist.get_rank() work.
    dist.init_process_group(
        "xla", rank=xla_model.get_ordinal(), world_size=xla_model.xrt_world_size(), init_method='pjrt://'
    )


def is_xla() -> bool:
    return bool(os.environ.get("PJRT_DEVICE"))


# TODO: Needed?
def is_distributed_xla() -> bool:
    # xrt_world_size works before dist.init_process_group
    return is_xla() and xla_model.xrt_world_size() > 1  # TODO: Or check dist init pjrt:// ?

def is_xla_master():
    if not is_xla():
        return False
    else:
        return xla_model.get_ordinal() == 0