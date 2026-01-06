import torch
from dreamer.utils.replay import Buffer


def test_buffer(tmpdir):
    path = tmpdir.mkdir("storage").join("buffer")
    buffer = Buffer(
        1000, ["obs", "is_first", "is_last"], path, load_existing=False, save_every=2
    )
    transition = {
        "obs": torch.randn(64, 64, 3),
        "is_first": torch.tensor([1.0]),
        "is_last": torch.tensor([0.0]),
    }
    buffer.add(transition)
    transition = {
        "obs": torch.randn(64, 64, 3),
        "is_first": torch.tensor([0.0]),
        "is_last": torch.tensor([1.0]),
    }
    buffer.add(transition)

    s = buffer.sample(16, 64)
    assert s.shape == (16, 64)
    assert len(buffer._buffer) == 2
    # Check file exists
    assert path.check()
