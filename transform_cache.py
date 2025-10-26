import re
from typing import Iterable
from torchvision import transforms
from data_utils import JPEGCompressionTransform


def _sanitize_component(value) -> str:
    """Convert a value into a filesystem-friendly string."""
    if isinstance(value, (list, tuple)):
        return "x".join(_sanitize_component(v) for v in value)

    value_str = str(value)
    value_str = value_str.replace(" ", "")
    value_str = value_str.replace("(", "").replace(")", "")
    value_str = value_str.replace("[", "").replace("]", "")
    value_str = value_str.replace(",", "_")
    # Replace dots to keep floating point information while avoiding directories
    value_str = value_str.replace(".", "p")
    # Collapse any remaining unsupported characters
    value_str = re.sub(r"[^0-9a-zA-Z_\-]", "", value_str)
    return value_str


def _describe_transform(transform) -> str:
    """Return a stable, lowercase string that describes a transform."""
    if transform is None:
        return ""

    # Handle Compose by concatenating the descriptions of each transform
    if isinstance(transform, transforms.Compose):
        parts: Iterable[str] = (
            desc for desc in (_describe_transform(t) for t in transform.transforms) if desc
        )
        return "__".join(parts)

    if isinstance(transform, JPEGCompressionTransform):
        return f"jpeg_q{_sanitize_component(transform.quality)}"

    if isinstance(transform, transforms.GaussianBlur):
        return "gaussianblur_k{ks}_s{sigma}".format(
            ks=_sanitize_component(transform.kernel_size),
            sigma=_sanitize_component(transform.sigma),
        )

    return ""


def build_transform_cache_suffix(transform) -> str:
    """Return a suffix that uniquely identifies a transform pipeline.

    The suffix is safe to append to filenames/directories and remains empty
    when no transform or only default behavior is provided.
    """
    description = _describe_transform(transform)
    if not description:
        return ""
    return f"__{description}"
