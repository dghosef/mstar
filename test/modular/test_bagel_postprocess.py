"""BAGEL's image response: the shared PNG writer must decode to the pixels the old PIL path
sent, and a format it cannot write fails at intake."""
import io

import numpy as np
import pytest
import torch

from mstar.model.bagel.bagel_model import BagelModel

PIL = pytest.importorskip("PIL.Image")


def test_image_png_decodes_to_the_truncated_pixels():
    g = torch.Generator().manual_seed(0)
    output = torch.rand(1, 3, 48, 64, generator=g)
    model = BagelModel.__new__(BagelModel)
    png = model.postprocess(output, "image", {})
    decoded = np.asarray(PIL.open(io.BytesIO(png)).convert("RGB"))
    # the old path: permute to HWC, scale, truncate to uint8
    expected = (output[0].permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    assert np.array_equal(decoded, expected)


def test_unsupported_output_format_fails_at_intake():
    model = BagelModel.__new__(BagelModel)
    with pytest.raises(ValueError, match="output_format"):
        model.process_prompt(None, [], ["image"], output_format="gif")


@pytest.mark.parametrize("size", [{"width": 8}, {"height": 15}, {"width": 0}, {"height": True}, {"width": "512"}])
def test_an_image_under_one_latent_patch_fails_at_intake(size):
    """Under 16 pixels a side, the image is no latent patch at all, and the
    request would otherwise fail deep in the denoise loop."""
    model = BagelModel.__new__(BagelModel)
    with pytest.raises(ValueError, match="width|height"):
        model.process_prompt(None, [], ["image"], **size)
    model.process_prompt(None, [], ["image"], width=16, height=512)
