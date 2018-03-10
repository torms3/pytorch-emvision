import collections


def _ntuple(n):
    """
    Copied from the PyTorch source code (https://github.com/pytorch).
    """
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_triple = _ntuple(3)


def pad_size(kernel_size, mode):
    assert mode in ['valid', 'same', 'full']
    ks = _triple(kernel_size)
    if mode == 'valid':
        return _triple(0)
    elif mode == 'same':
        assert all([x % 2 for x in ks])
        return tuple(x // 2 for x in ks)
    elif mode == 'full':
        return tuple(x - 1 for x in ks)
