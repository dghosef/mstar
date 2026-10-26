"""The stop check's per-rid host dicts are built only when something reads
them."""
import torch

from mstar.worker.worker import Worker, _LazyHostOutputs, _LazyPerRid


def test_lazy_per_rid_builds_on_first_read_only():
    host = {"new_token": torch.arange(3).view(3, 1), "flag": torch.zeros(3, dtype=torch.bool)}
    rids = [7, 8, 9]
    lazy = _LazyPerRid(host, rids)
    assert lazy._built is None
    assert len(lazy) == 3 and bool(lazy)
    assert lazy._built is None, "len and truth must not build the dicts"
    eager = Worker._rows_to_per_rid(host, rids)
    assert lazy.get(8).keys() == eager[8].keys()
    for name in eager[8]:
        assert torch.equal(lazy.get(8)[name][0], eager[8][name][0])
    assert lazy.get(42) is None
    assert 7 in lazy and list(lazy) == rids
    assert lazy[9]["new_token"][0].item() == 2
    assert not _LazyPerRid(host, [])


def test_lazy_host_outputs_copy_only_the_names_read():
    """A stop check that reads the token copies the token alone, and one that
    reads nothing (an iteration-count stop) copies nothing."""
    source = {
        1: {"new_token": [torch.tensor([5])], "latents": [torch.ones(4, 4)]},
        2: {"new_token": [torch.tensor([6])], "latents": [torch.ones(4, 4)]},
        3: "not-a-dict",
    }
    copied = []

    def copy_name(name):
        copied.append(name)
        return {rid: [t.clone() for t in row[name]] for rid, row in source.items()
                if isinstance(row, dict) and name in row}

    lazy = _LazyHostOutputs(source, copy_name)
    row = lazy.get(1)
    assert row and "latents" in row and len(row) == 2 and set(row) == {"new_token", "latents"}
    assert copied == [], "presence, length and iteration must not copy"
    assert row["new_token"][0].item() == 5
    assert lazy[2]["new_token"][0].item() == 6
    assert copied == ["new_token"], "a name is copied once, for every row"
    assert lazy.get(3) == "not-a-dict" and lazy.get(4) is None
    assert set(lazy.landed()) == {1, 2}
    assert all(set(names) == {"new_token"} for names in lazy.landed().values())
