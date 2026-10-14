"""Person crop -> L2-normalised ReID embedding using pretrained OSNet."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from reid._osnet import OSNet

# Preprocessing used by torchreid when these weights were trained.
INPUT_HEIGHT = 256
INPUT_WIDTH = 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    num_classes: int


# Official torchreid model zoo, "cross-domain" table: OSNet-x1.0 trained on all of MSMT17.
# Chosen over the Market1501/Duke models because our cameras are an unseen domain.
DEFAULT_MODEL = ModelSpec(
    name="osnet_x1_0_msmt17",
    url="https://drive.usercontent.google.com/download?id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x&export=download&confirm=t",
    sha256="48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727",
    num_classes=4101,
)


def default_weights_dir() -> Path:
    """Weights cache location; override with the REID_WEIGHTS_DIR env var."""
    env = os.environ.get("REID_WEIGHTS_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "ai_perception_engine" / "reid"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights(spec: ModelSpec = DEFAULT_MODEL, weights_dir: Path | None = None) -> Path:
    """Return a local path to the checkpoint, downloading and verifying it if missing."""
    weights_dir = weights_dir or default_weights_dir()
    path = weights_dir / f"{spec.name}.pt"
    if path.exists() and _sha256(path) == spec.sha256:
        return path

    weights_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.part")
    print(f"[reid] downloading {spec.name} weights to {path}")
    urllib.request.urlretrieve(spec.url, tmp)
    digest = _sha256(tmp)
    if digest != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {spec.name}: expected {spec.sha256}, got {digest}")
    tmp.replace(path)
    return path


ImageInput = str | os.PathLike | np.ndarray


def _load_rgb(image: ImageInput) -> Image.Image:
    """Load an image path (any PIL format) or an OpenCV-style BGR uint8 HxWx3 array as RGB."""
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected a BGR image array of shape (H, W, 3), got {image.shape}")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"Image array is empty: shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image array, got dtype {image.dtype}")
        return Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))

    if isinstance(image, (str, os.PathLike)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError(f"Not a readable image file: {path}") from exc

    raise TypeError(f"Unsupported image input type: {type(image).__name__}")


def _preprocess(img: Image.Image) -> torch.Tensor:
    img = img.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class Embedder:
    """Wraps a pretrained OSNet ReID model.

    Usage:
        embedder = Embedder()
        vec = embedder.embed("crop.jpg")   # np.ndarray, shape (512,), float32, unit L2 norm
    """

    def __init__(
        self,
        weights_path: str | os.PathLike | None = None,
        device: str | None = None,
        spec: ModelSpec = DEFAULT_MODEL,
    ):
        self.model_name = spec.name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights_path = Path(weights_path) if weights_path else ensure_weights(spec)

        model = OSNet(num_classes=spec.num_classes)
        state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(self.device).eval()
        self.embedding_dim = model.feature_dim

    def embed(self, image: ImageInput) -> np.ndarray:
        """Return a 1-D float32 embedding of length `embedding_dim`, L2-normalised to unit norm.

        `image` is a file path or an OpenCV-style BGR uint8 array of a single person crop.
        """
        batch = _preprocess(_load_rgb(image)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model(batch)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features[0].cpu().numpy().astype(np.float32)
