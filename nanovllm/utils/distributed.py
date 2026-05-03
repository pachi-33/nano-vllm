import torch.distributed as dist

_tp_group = None


def get_tp_group():
    return _tp_group


def set_tp_group(group):
    global _tp_group
    _tp_group = group


def tp_world_size():
    g = _tp_group
    if g is not None:
        return dist.get_world_size(g)
    return dist.get_world_size()


def tp_rank():
    g = _tp_group
    if g is not None:
        return dist.get_rank(g)
    return dist.get_rank()
