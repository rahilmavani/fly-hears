import asyncio
import io
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from app.server import create_app

app = create_app(Path('/tmp/unused-fly-test-run'))


async def get(path, application=app):
    """Exercise the ASGI route and actual response bytes without an HTTP client dependency."""
    messages = []

    async def receive():
        await asyncio.Future()

    async def send(message):
        messages.append(message)

    await application({"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
               "http_version": "1.1", "method": "GET", "scheme": "http",
               "path": path, "raw_path": path.encode(), "query_string": b"",
               "headers": [], "server": ("localhost", 8000), "client": ("localhost", 1234)},
              receive, send)
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], dict(start["headers"]), body


class SampleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.files = []
        for digit in range(10):
            path = Path(directory.name) / f"{digit}_speaker_0.wav"
            audio = io.BytesIO()
            with wave.open(audio, "wb") as wav:
                wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                wav.writeframes(struct.pack("<h", digit * 100) * 800)
            path.write_bytes(audio.getvalue())
            self.files.append(path)
        previous = app.state.samples
        app.state.samples = self.files
        self.addCleanup(setattr, app.state, 'samples', previous)

    async def test_every_label_url_and_audio_file_match(self):
        status, headers, body = await get("/samples")
        self.assertEqual(status, 200)
        self.assertEqual(headers[b"cache-control"], b"no-store")
        samples = json.loads(body)
        self.assertEqual([s["digit"] for s in samples], list(range(10)))
        for sample in samples:
            status, headers, audio = await get(sample["url"])
            self.assertEqual(status, 200)
            self.assertEqual(headers[b"cache-control"], b"no-store")
            self.assertEqual(audio, self.files[sample["digit"]].read_bytes())
            self.assertTrue(sample["name"].startswith(f"{sample['digit']}_"))

    async def test_old_page_url_cannot_change_digits_after_list_changes(self):
        _, _, body = await get("/samples")
        digit_four = json.loads(body)[4]
        app.state.samples = list(reversed(self.files))
        status, _, audio = await get(digit_four["url"])
        self.assertEqual(status, 200)
        self.assertEqual(audio, self.files[4].read_bytes())
        app.state.samples = [self.files[0]]
        status, _, _ = await get(digit_four["url"])
        self.assertEqual(status, 404)

    async def test_ambiguous_numbered_links_require_a_refresh(self):
        status, headers, _ = await get("/sample/4")
        self.assertEqual(status, 410)
        self.assertEqual(headers[b"cache-control"], b"no-store")


if __name__ == "__main__":
    unittest.main()
