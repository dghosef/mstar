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
    """Every image-output request passes the API server's submit, so a format the shared
    writer can't produce fails there, for BAGEL and the DiT image models alike."""
    from mstar.api_server.entrypoint import APIServer

    server = APIServer.__new__(APIServer)
    with pytest.raises(ValueError, match="output_format"):
        server.submit_request(
            text="a cat", input_modalities=["text"], output_modalities=["image"],
            model_kwargs={"output_format": "gif"}, streaming=False,
        )
