"""Video decoding and encoding for frame-wise LocateAnything inference."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    source_fps: float
    duration_seconds: float
    frame_count: int
    frame_count_exact: bool
    rotation_degrees: int
    has_audio: bool

    @property
    def output_fps(self) -> float:
        return self.source_fps if self.source_fps > 0 else 5.0


def create_video_output_dir(
    layout_root: Path,
    video_path: Path,
    output_dir: Path | None = None,
    *,
    sequence: int | None = None,
) -> Path:
    """Create one self-contained output directory for a video request."""

    if output_dir is not None:
        index = f"{sequence:04d}" if sequence is not None else "run"
        suffix = f"video_{index}_{uuid.uuid4().hex[:8]}"
        root = output_dir.expanduser().resolve() / suffix
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("._")
        name = name or "video"
        root = (
            layout_root
            / "outputs"
            / "video"
            / f"{timestamp}_{name}_{uuid.uuid4().hex[:8]}"
        )
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    return root


def _require_program(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(
            f"{name} is required for video inference; install it with "
            "'sudo apt-get install -y ffmpeg'"
        )
    return executable


def _run(command: list[str], action: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"failed to {action}: {details}")
    return completed


def _parse_rate(value: object) -> float:
    try:
        rate = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return 0.0
    return rate if rate > 0 else 0.0


def probe_video(video_path: Path) -> VideoInfo:
    """Read the first video stream without decoding the video."""

    ffprobe = _require_program("ffprobe")
    completed = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration"
            ":stream_side_data=rotation:format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        "read video metadata",
    )
    try:
        payload = json.loads(completed.stdout)
        streams = payload["streams"]
        stream = next(item for item in streams if item.get("codec_type") == "video")
        coded_width = int(stream["width"])
        coded_height = int(stream["height"])
    except (
        json.JSONDecodeError,
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
    ) as error:
        raise RuntimeError(f"video has no readable video stream: {video_path}") from error

    rotation_degrees = 0
    for side_data in stream.get("side_data_list", []):
        try:
            rotation_degrees = int(side_data.get("rotation", 0)) % 360
        except (TypeError, ValueError):
            continue
    if rotation_degrees in {90, 270}:
        width, height = coded_height, coded_width
    else:
        width, height = coded_width, coded_height
    has_audio = any(item.get("codec_type") == "audio" for item in streams)

    source_fps = _parse_rate(stream.get("avg_frame_rate"))
    if source_fps == 0.0:
        source_fps = _parse_rate(stream.get("r_frame_rate"))
    duration_seconds = 0.0
    for raw_duration in (
        stream.get("duration"),
        payload.get("format", {}).get("duration"),
    ):
        try:
            duration_seconds = float(raw_duration)
        except (TypeError, ValueError):
            continue
        if duration_seconds > 0:
            break
    if duration_seconds <= 0:
        raise RuntimeError(f"video duration is unavailable or invalid: {video_path}")

    frame_count_exact = True
    try:
        frame_count = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        frame_count = 0
    if frame_count <= 0 and source_fps > 0:
        frame_count = int(round(duration_seconds * source_fps))
        frame_count_exact = False
    if frame_count <= 0:
        raise RuntimeError(f"video frame count is unavailable or invalid: {video_path}")
    return VideoInfo(
        width=width,
        height=height,
        source_fps=source_fps,
        duration_seconds=duration_seconds,
        frame_count=frame_count,
        frame_count_exact=frame_count_exact,
        rotation_degrees=rotation_degrees,
        has_audio=has_audio,
    )


class VideoFrameReader:
    """Stream decoded RGB frames without materializing the complete video."""

    def __init__(self, video_path: Path, info: VideoInfo) -> None:
        self.video_path = video_path
        self.info = info
        self._process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "VideoFrameReader":
        ffmpeg = _require_program("ffmpeg")
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(self.video_path),
                "-map",
                "0:v:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return self

    def __iter__(self) -> Iterator[bytes]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("video decoder is not running")
        frame_size = self.info.width * self.info.height * 3
        while True:
            data = bytearray()
            while len(data) < frame_size:
                chunk = self._process.stdout.read(frame_size - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            if not data:
                return
            if len(data) != frame_size:
                raise RuntimeError(
                    f"video decoder returned a partial frame: {len(data)}/{frame_size} bytes"
                )
            yield bytes(data)

    def __exit__(self, error_type, _error, _traceback) -> None:
        process = self._process
        if process is None:
            return
        if error_type is not None and process.poll() is None:
            process.terminate()
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
        stderr = process.stderr.read() if process.stderr is not None else b""
        if error_type is None and return_code != 0:
            raise RuntimeError(
                "failed to decode video: " + stderr.decode("utf-8", "replace").strip()
            )


class VideoFrameWriter:
    """Stream annotated RGB frames into an H.264 MP4."""

    codec = "h264"

    def __init__(
        self,
        output_path: Path,
        source_video: Path,
        info: VideoInfo,
    ) -> None:
        self.output_path = output_path
        self.source_video = source_video
        self.info = info
        self._process: subprocess.Popen[bytes] | None = None
        self._video_only_path = output_path.with_name(output_path.stem + ".video-only.mp4")
        self.audio_preserved = False
        self.warning: str | None = None

    def __enter__(self) -> "VideoFrameWriter":
        ffmpeg = _require_program("ffmpeg")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{self.info.width}x{self.info.height}",
                "-r",
                f"{self.info.output_fps:.12g}",
                "-i",
                "pipe:0",
                "-an",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self._video_only_path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return self

    def write(self, rgb_frame: bytes) -> None:
        expected_size = self.info.width * self.info.height * 3
        if len(rgb_frame) != expected_size:
            raise ValueError(
                f"annotated frame has {len(rgb_frame)} bytes; expected {expected_size}"
            )
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("video encoder is not running")
        self._process.stdin.write(rgb_frame)

    def __exit__(self, error_type, _error, _traceback) -> None:
        process = self._process
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if error_type is not None and process.poll() is None:
            process.terminate()
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
        stderr = process.stderr.read() if process.stderr is not None else b""
        if error_type is not None:
            self._video_only_path.unlink(missing_ok=True)
            self.output_path.unlink(missing_ok=True)
        elif return_code != 0:
            self._video_only_path.unlink(missing_ok=True)
            self.output_path.unlink(missing_ok=True)
            raise RuntimeError(
                "failed to encode annotated video: "
                + stderr.decode("utf-8", "replace").strip()
            )
        elif not self.info.has_audio:
            self._video_only_path.replace(self.output_path)
        else:
            ffmpeg = _require_program("ffmpeg")
            try:
                _run(
                    [
                        ffmpeg,
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(self._video_only_path),
                        "-i",
                        str(self.source_video),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c",
                        "copy",
                        "-shortest",
                        "-movflags",
                        "+faststart",
                        str(self.output_path),
                    ],
                    "preserve source audio",
                )
            except RuntimeError as error:
                self.warning = str(error)
                self._video_only_path.replace(self.output_path)
            else:
                self.audio_preserved = True
                self._video_only_path.unlink(missing_ok=True)
