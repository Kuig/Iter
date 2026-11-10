"""Image files for image nodes: locating local ones and downloading remote ones.

Used by Phase 2 when image_handling is 'standalone_nodes', to attach each image
file to its AI call. This holds the only HTTP code in Iter: UnifiedAiClient
attaches local files only, so a remote image is downloaded first.
"""
from __future__ import annotations

import mimetypes
import urllib.parse
import urllib.request
from pathlib import Path

import itertool.logger as logger
from itertool.markdown_utils import image_source
from itertool.models import Node


def fetch_image(url: str, dest_dir: Path, index: int) -> Path:
    """Download a remote image into dest_dir.

    The file keeps an image extension, taken from the URL path or else from the
    Content-Type, because UnifiedAiClient decides how to send a file by its
    extension.

    Args:
        url: http(s) URL of the image.
        dest_dir: Directory the file is written to.
        index: Node.index of the image, used to name the file.

    Returns:
        Path of the downloaded file.

    Raises:
        urllib.error.URLError: If the download fails.
    """
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
        content_type = response.headers.get_content_type()
    suffix = Path(urllib.parse.urlparse(url).path).suffix
    if not mimetypes.guess_type(f"x{suffix}")[0]:
        suffix = mimetypes.guess_extension(content_type) or ""
    path = dest_dir / f"image-{index}{suffix}"
    path.write_bytes(data)
    return path


def resolve_images(nodes: list[Node], base_dir: Path, dest_dir: Path) -> dict[int, Path]:
    """Locate the file of every image node, downloading remote ones.

    Runs before any node is sent to the AI, so a missing image stops the run
    before it spends calls on the nodes in front of it.

    Args:
        nodes: The node list after the pre-passes.
        base_dir: Directory relative image paths are resolved against.
        dest_dir: Directory downloaded images are written to.

    Returns:
        Image file paths, by Node.index.

    Raises:
        FileNotFoundError: If a local image does not exist.
        urllib.error.URLError: If a remote image cannot be downloaded.
    """
    paths: dict[int, Path] = {}
    for node in nodes:
        if node.node_type != "image":
            continue
        src = image_source(node.content)
        if urllib.parse.urlparse(src).scheme in ("http", "https"):
            logger.log_action(f"Downloading image of node §{node.index}: {src}")
            paths[node.index] = fetch_image(src, dest_dir, node.index)
            continue
        path = base_dir / urllib.parse.unquote(src)
        if not path.is_file():
            raise FileNotFoundError(f"Image of node §{node.index} not found: {path}")
        paths[node.index] = path
    return paths
