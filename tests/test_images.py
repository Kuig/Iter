"""Tests for locating and downloading the files of image nodes.

No test in this suite reaches the network: urlopen is stubbed.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import itertool.images as images
import itertool.logger as logger
from itertool.models import Node


def setUpModule():
    logger.set_backend("silent")


def tearDownModule():
    logger.set_backend("console")


class _FakeResponse:
    """What urllib.request.urlopen returns, reduced to what fetch_image reads."""

    def __init__(self, data: bytes, content_type: str):
        self._data = data
        self.headers = self
        self._content_type = content_type

    def get_content_type(self):
        return self._content_type

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _image(index: int, src: str) -> Node:
    return Node(index=index, node_type="image", content=f"![c]({src})")


class TestResolveImages(unittest.TestCase):
    """A source becomes a file, relative to the document or downloaded."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.downloads = Path(tempfile.mkdtemp())
        (self.dir / "pics").mkdir()
        self.image = self.dir / "pics" / "chart.png"
        self.image.write_bytes(b"png")

    def _resolve(self, *nodes: Node) -> dict[int, Path]:
        return images.resolve_images(list(nodes), self.dir, self.downloads)

    def test_relative_path_is_resolved_against_base_dir(self):
        self.assertEqual(self._resolve(_image(2, "pics/chart.png")), {2: self.image})

    def test_source_with_spaces_and_title(self):
        spaced = self.dir / "pics" / "my chart.png"
        spaced.write_bytes(b"png")
        node = Node(index=1, node_type="image", content='![c](<pics/my chart.png> "Title")')
        self.assertEqual(self._resolve(node), {1: spaced})

    def test_percent_encoded_path(self):
        spaced = self.dir / "pics" / "my chart.png"
        spaced.write_bytes(b"png")
        self.assertEqual(self._resolve(_image(1, "pics/my%20chart.png")), {1: spaced})

    def test_other_nodes_are_ignored(self):
        text = Node(index=1, node_type="text", content="![c](pics/missing.png) is mentioned here")
        self.assertEqual(self._resolve(text), {})

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            self._resolve(_image(4, "pics/missing.png"))
        self.assertIn("§4", str(ctx.exception))

    def test_url_extension_comes_from_the_content_type_when_the_path_has_none(self):
        response = _FakeResponse(b"remote", "image/jpeg")
        with patch.object(images.urllib.request, "urlopen", return_value=response):
            paths = self._resolve(_image(3, "https://example.com/render?id=1"))
        self.assertEqual(paths[3].parent, self.downloads)
        self.assertIn(paths[3].suffix, (".jpg", ".jpeg", ".jpe"))
        self.assertEqual(paths[3].read_bytes(), b"remote")

    def test_url_extension_comes_from_the_path(self):
        response = _FakeResponse(b"remote", "application/octet-stream")
        with patch.object(images.urllib.request, "urlopen", return_value=response):
            paths = self._resolve(_image(3, "https://example.com/a/chart.webp"))
        self.assertEqual(paths[3].suffix, ".webp")


if __name__ == "__main__":
    unittest.main()
