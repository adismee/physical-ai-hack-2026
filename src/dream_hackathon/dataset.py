"""A LeRobot v2.1 dataset writer: the format Dream fine-tuning reads (no LeRobot install needed).

One dataset is one folder:

```
meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl, meta/episodes_stats.jsonl
data/chunk-000/episode_000000.parquet
videos/chunk-000/<video key>/episode_000000.mp4
```

Each episode's parquet holds `observation.state`, `action`, `timestamp`, `frame_index`, `episode_index`, `index`
and `task_index`; each camera's video is H.264, one frame per row, at the dataset's frame rate. A new session adds
episodes to an existing folder. While an episode records, its videos are encoded on one thread per camera into a
scratch folder outside the dataset, so a crash never leaves a half-written file inside it.

@see DatasetWriter, Episode
"""

import dataclasses
import json
import os
import pathlib
import queue
import shutil
import tempfile
import threading
from typing import Any

import av
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v2.1"
## Episodes per `chunk-<NNN>` folder, as LeRobot writes.
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
## The shortest episode Dream trains on: 1.6 s from first frame to last.
MIN_EPISODE_S = 1.6
## Every this many frames, one is sampled into the episode's image statistics.
_IMAGE_STATS_EVERY = 10
_SCRATCH_ROOT = pathlib.Path("~/.cache/dream_hackathon/recording").expanduser()


class DatasetError(RuntimeError):
    """The dataset folder cannot take this recording (another frame rate, other cameras or widths)."""


def min_frames(fps: int) -> int:
    """The fewest frames an episode may hold at `fps`: 49 at 30 fps."""
    return round(MIN_EPISODE_S * fps) + 1


class DatasetWriter:
    """Adds episodes to one LeRobot v2.1 dataset folder, creating it on first use.

    Attributes:
    - root: The dataset folder.
    - fps: Frames per second.
    - total_episodes: Episodes saved so far, this session's included.
    """

    def __init__(
        self,
        root: pathlib.Path,
        fps: int,
        robot_type: str,
        state_names: list[str],
        video_keys: list[str],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Opens the folder, or prepares a new one; nothing is written until the first episode is saved.

        @param root The dataset folder.
        @param fps Frames per second.
        @param robot_type Recorded in `info.json`, e.g. `"yam"`.
        @param state_names One name per value of the state and action vectors.
        @param video_keys The cameras' keys, e.g. `observation.images.overhead`.
        @param frame_width, frame_height The size every frame is recorded at.
        @throws DatasetError The folder holds a dataset recorded another way.
        """
        self.root = root
        self.fps = fps
        self._video_keys = list(video_keys)
        self._frame_size = (frame_width, frame_height)
        self._features = _features(state_names, self._video_keys, frame_width, frame_height, fps)
        info_path = root / "meta" / "info.json"
        if info_path.exists():
            self._info = json.loads(info_path.read_text())
            _check_compatible(root, self._info, fps, self._features)
            self._tasks = _read_tasks(root)
        else:
            self._info = _new_info(robot_type, fps, self._features, len(self._video_keys))
            self._tasks = {}
        self.total_episodes = int(self._info["total_episodes"])

    def start_episode(self, task: str) -> "Episode":
        """Starts recording one episode.

        @param task The task text the episode is trained on; write it the way you will prompt the robot.
        @return The episode; add frames, then save() or discard() it.
        """
        if not task.strip():
            raise DatasetError("an episode needs a task text")
        return Episode(self, task.strip())

    def _commit(self, episode: "Episode", table: pa.Table, stats: dict[str, Any], scratch: pathlib.Path) -> int:
        """Moves a finished episode into the dataset and updates `meta/`; returns its index."""
        index = self.total_episodes
        chunk = index // CHUNKS_SIZE
        is_new_task = episode.task not in self._tasks
        task_index = self._tasks.setdefault(episode.task, len(self._tasks))
        frames = table.num_rows
        first_index = int(self._info["total_frames"])
        table = table.append_column("episode_index", pa.array(np.full(frames, index, dtype=np.int64)))
        table = table.append_column("index", pa.array(np.arange(first_index, first_index + frames, dtype=np.int64)))
        table = table.append_column("task_index", pa.array(np.full(frames, task_index, dtype=np.int64)))
        stats["episode_index"] = _scalar_stats(np.full(frames, index))
        stats["index"] = _scalar_stats(np.arange(first_index, first_index + frames))
        stats["task_index"] = _scalar_stats(np.full(frames, task_index))

        data_path = self.root / DATA_PATH.format(episode_chunk=chunk, episode_index=index)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path)
        for key in self._video_keys:
            video_path = self.root / VIDEO_PATH.format(episode_chunk=chunk, video_key=key, episode_index=index)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(scratch / f"{key}.mp4", video_path)

        meta = self.root / "meta"
        if is_new_task:
            _append_jsonl(meta / "tasks.jsonl", {"task_index": task_index, "task": episode.task})
        _append_jsonl(meta / "episodes.jsonl", {"episode_index": index, "tasks": [episode.task], "length": frames})
        _append_jsonl(meta / "episodes_stats.jsonl", {"episode_index": index, "stats": stats})
        self.total_episodes = index + 1
        self._info["total_episodes"] = self.total_episodes
        self._info["total_frames"] = first_index + frames
        self._info["total_tasks"] = len(self._tasks)
        self._info["total_videos"] = self.total_episodes * len(self._video_keys)
        self._info["total_chunks"] = chunk + 1
        self._info["splits"] = {"train": f"0:{self.total_episodes}"}
        _write_json(meta / "info.json", self._info)
        return index


class Episode:
    """One episode being recorded: add_frame() each tick, then save() or discard().

    Attributes:
    - task: The episode's task text.
    - frames: Frames added so far.
    """

    def __init__(self, writer: DatasetWriter, task: str) -> None:
        """Opens one video encoder thread per camera in a fresh scratch folder."""
        self.task = task
        self.frames = 0
        self._writer = writer
        self._states: list[npt.NDArray[np.float32]] = []
        self._actions: list[npt.NDArray[np.float32]] = []
        _SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        self._scratch = pathlib.Path(tempfile.mkdtemp(dir=_SCRATCH_ROOT))
        self._encoders = {
            key: _VideoEncoder(self._scratch / f"{key}.mp4", writer.fps, *writer._frame_size)
            for key in writer._video_keys
        }

    def add_frame(
        self,
        state: npt.NDArray[np.float32],
        action: npt.NDArray[np.float32],
        images: dict[str, npt.NDArray[np.uint8]],
    ) -> None:
        """Adds one frame.

        @param state The measured state vector.
        @param action The commanded action vector.
        @param images Per video key, a BGR `[height, width, 3]` frame at the dataset's frame size.
        """
        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(np.asarray(action, dtype=np.float32))
        for key, encoder in self._encoders.items():
            encoder.put(images[key])
        self.frames += 1

    def save(self) -> int:
        """Finishes the videos and adds the episode to the dataset.

        @return The episode's index.
        @throws DatasetError The episode is shorter than Dream trains on (min_frames()); it is discarded.
        """
        stats = {key: encoder.finish() for key, encoder in self._encoders.items()}
        fps = self._writer.fps
        if self.frames < min_frames(fps):
            self._cleanup()
            raise DatasetError(
                f"episode too short: {self.frames} frames, at least {min_frames(fps)} ({MIN_EPISODE_S} s) needed"
            )
        states = np.stack(self._states)
        actions = np.stack(self._actions)
        frame_index = np.arange(self.frames, dtype=np.int64)
        timestamps = (frame_index / fps).astype(np.float32)
        width = states.shape[1]
        table = pa.table(
            {
                "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(states.reshape(-1)), width),
                "action": pa.FixedSizeListArray.from_arrays(pa.array(actions.reshape(-1)), actions.shape[1]),
                "timestamp": pa.array(timestamps),
                "frame_index": pa.array(frame_index),
            }
        )
        stats.update(
            {
                "observation.state": _vector_stats(states),
                "action": _vector_stats(actions),
                "timestamp": _scalar_stats(timestamps),
                "frame_index": _scalar_stats(frame_index),
            }
        )
        index = self._writer._commit(self, table, stats, self._scratch)
        self._cleanup()
        return index

    def discard(self) -> None:
        """Drops the episode; nothing reaches the dataset."""
        for encoder in self._encoders.values():
            encoder.finish()
        self._cleanup()

    def _cleanup(self) -> None:
        shutil.rmtree(self._scratch, ignore_errors=True)


class _VideoEncoder:
    """One camera's H.264 encoder, fed from a queue by its own thread, sampling frames for the image statistics."""

    def __init__(self, path: pathlib.Path, fps: int, width: int, height: int) -> None:
        self._size = (width, height)
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        # A keyframe every 10 frames keeps random access fast for the trainer's decoder.
        self._stream.options = {"preset": "veryfast", "crf": "23", "g": "10"}
        self._queue: queue.Queue[npt.NDArray[np.uint8] | None] = queue.Queue(maxsize=10 * fps)
        self._samples: list[npt.NDArray[np.float32]] = []
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=f"encode-{path.stem}", daemon=True)
        self._thread.start()

    def put(self, bgr: npt.NDArray[np.uint8]) -> None:
        if self._error is not None:
            raise DatasetError(f"video encoder failed: {self._error}")
        if bgr.shape[1::-1] != self._size:
            raise DatasetError(f"frame is {bgr.shape[1]}x{bgr.shape[0]}; this dataset records {self._size}")
        self._queue.put(bgr)

    def finish(self) -> dict[str, Any]:
        """Flushes and closes the file; returns the image statistics LeRobot keeps per episode."""
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise DatasetError(f"video encoder failed: {self._error}")
        if not self._samples:
            return {}
        # LeRobot's image statistics: per RGB channel, over every pixel of the sampled frames, as [3, 1, 1].
        pixels = np.concatenate([sample.reshape(-1, 3) for sample in self._samples])
        stats = {
            "min": pixels.min(axis=0),
            "max": pixels.max(axis=0),
            "mean": pixels.mean(axis=0),
            "std": pixels.std(axis=0),
        }
        out: dict[str, Any] = {name: value.reshape(3, 1, 1).tolist() for name, value in stats.items()}
        out["count"] = [len(self._samples)]
        return out

    def _run(self) -> None:
        count = 0
        try:
            while (bgr := self._queue.get()) is not None:
                frame = av.VideoFrame.from_ndarray(bgr, format="bgr24").reformat(format="yuv420p")
                frame.pts = count
                for packet in self._stream.encode(frame):
                    self._container.mux(packet)
                if count % _IMAGE_STATS_EVERY == 0:
                    # BGR to RGB, scaled to [0, 1], on a quarter-size copy (statistics, not pixels).
                    self._samples.append(bgr[::4, ::4, ::-1].astype(np.float32) / 255.0)
                count += 1
            for packet in self._stream.encode():
                self._container.mux(packet)
        except BaseException as error:  # noqa: BLE001 - reported to the recording thread on its next call
            self._error = error
            while self._queue.get() is not None:
                pass
        finally:
            self._container.close()


@dataclasses.dataclass(frozen=True)
class EpisodeInfo:
    """One saved episode, as `meta/episodes.jsonl` lists it.

    Attributes:
    - index: The episode's index.
    - frames: Its length in frames.
    - tasks: Its task texts.
    """

    index: int
    frames: int
    tasks: tuple[str, ...]


def list_episodes(root: pathlib.Path) -> list[EpisodeInfo]:
    """The dataset's episodes, in order; empty when the folder holds no dataset yet."""
    path = root / "meta" / "episodes.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [EpisodeInfo(int(row["episode_index"]), int(row["length"]), tuple(row["tasks"])) for row in rows]


def video_keys_of(root: pathlib.Path) -> list[str]:
    """The camera video keys a dataset records, from its `meta/info.json`."""
    info = json.loads((root / "meta" / "info.json").read_text())
    return sorted(key for key, feature in info["features"].items() if feature.get("dtype") == "video")


def drop_episodes(root: pathlib.Path, drop: set[int]) -> int:
    """Deletes episodes from a dataset and renumbers the ones after them, as Dream requires (0, 1, 2, ...).

    Each later episode's parquet is rewritten with its new `episode_index` and the dataset-wide frame `index`, and
    its videos are renamed; `meta/` is rewritten to match. Do not run it while a collector records into the folder.

    @param root The dataset folder.
    @param drop The indices of the episodes to delete.
    @return How many episodes are left.
    @throws DatasetError An index is not one of the dataset's episodes.
    """
    episodes = list_episodes(root)
    unknown = sorted(drop - {episode.index for episode in episodes})
    if unknown:
        raise DatasetError(f"{root} has no episode {unknown[0]} (it has 0 to {len(episodes) - 1})")
    keys = video_keys_of(root)
    meta = root / "meta"
    stats_rows = [json.loads(line) for line in (meta / "episodes_stats.jsonl").read_text().splitlines() if line]
    stats_by_episode = {int(row["episode_index"]): row["stats"] for row in stats_rows}

    def data_path(index: int) -> pathlib.Path:
        return root / DATA_PATH.format(episode_chunk=index // CHUNKS_SIZE, episode_index=index)

    def video_path(index: int, key: str) -> pathlib.Path:
        return root / VIDEO_PATH.format(episode_chunk=index // CHUNKS_SIZE, video_key=key, episode_index=index)

    for index in sorted(drop):
        data_path(index).unlink()
        for key in keys:
            video_path(index, key).unlink()
    kept: list[EpisodeInfo] = []
    kept_stats: list[dict[str, Any]] = []
    first_frame = 0
    for episode in episodes:
        if episode.index in drop:
            continue
        new = len(kept)
        # New indices only ever move down, onto a file already deleted or already moved, so nothing is overwritten.
        table = pq.read_table(data_path(episode.index))
        frames = np.arange(first_frame, first_frame + table.num_rows, dtype=np.int64)
        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array(np.full(table.num_rows, new, dtype=np.int64)),
        )
        table = table.set_column(table.schema.get_field_index("index"), "index", pa.array(frames))
        target = data_path(new)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
        if new != episode.index:
            data_path(episode.index).unlink()
            for key in keys:
                video_path(new, key).parent.mkdir(parents=True, exist_ok=True)
                os.replace(video_path(episode.index, key), video_path(new, key))
        stats = dict(stats_by_episode[episode.index])
        stats["episode_index"] = _scalar_stats(np.full(table.num_rows, new))
        stats["index"] = _scalar_stats(frames)
        kept.append(EpisodeInfo(new, episode.frames, episode.tasks))
        kept_stats.append({"episode_index": new, "stats": stats})
        first_frame += table.num_rows

    _write_lines(
        meta / "episodes.jsonl", [{"episode_index": e.index, "tasks": list(e.tasks), "length": e.frames} for e in kept]
    )
    _write_lines(meta / "episodes_stats.jsonl", kept_stats)
    info = json.loads((meta / "info.json").read_text())
    info["total_episodes"] = len(kept)
    info["total_frames"] = first_frame
    info["total_videos"] = len(kept) * len(keys)
    info["total_chunks"] = (len(kept) - 1) // CHUNKS_SIZE + 1 if kept else 0
    info["splits"] = {"train": f"0:{len(kept)}"}
    _write_json(meta / "info.json", info)
    return len(kept)


def _features(
    state_names: list[str],
    video_keys: list[str],
    width: int,
    height: int,
    fps: int,
) -> dict[str, Any]:
    """The `features` block of `info.json`."""
    vector = {"dtype": "float32", "shape": [len(state_names)], "names": list(state_names)}
    features: dict[str, Any] = {"action": dict(vector), "observation.state": dict(vector)}
    for key in video_keys:
        features[key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": fps,
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    for column in ("frame_index", "episode_index", "index", "task_index"):
        features[column] = {"dtype": "int64", "shape": [1], "names": None}
    return features


def _new_info(robot_type: str, fps: int, features: dict[str, Any], cameras: int) -> dict[str, Any]:
    """A new dataset's `info.json`, with no episodes yet."""
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": "0:0"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
    }


def _check_compatible(root: pathlib.Path, info: dict[str, Any], fps: int, features: dict[str, Any]) -> None:
    """Refuses to add episodes recorded another way to an existing dataset."""
    if info.get("codebase_version") != CODEBASE_VERSION:
        raise DatasetError(f"{root} is LeRobot {info.get('codebase_version')}, not {CODEBASE_VERSION}")
    if info.get("fps") != fps:
        raise DatasetError(f"{root} is recorded at {info.get('fps')} fps; this station records {fps}")
    for key in ("action", "observation.state"):
        if info["features"].get(key, {}).get("shape") != features[key]["shape"]:
            raise DatasetError(f"{root}: '{key}' is {info['features'].get(key)}, this station records {features[key]}")
    recorded = sorted(k for k, v in info["features"].items() if v.get("dtype") == "video")
    expected = sorted(k for k, v in features.items() if v.get("dtype") == "video")
    if recorded != expected:
        raise DatasetError(f"{root} has cameras {recorded}; this station records {expected}")
    for key in expected:
        if info["features"][key]["shape"] != features[key]["shape"]:
            raise DatasetError(
                f"{root}: {key} is {info['features'][key]['shape']}, this station records {features[key]['shape']}"
            )


def _read_tasks(root: pathlib.Path) -> dict[str, int]:
    """Task text to task index, from `meta/tasks.jsonl`."""
    path = root / "meta" / "tasks.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["task"]: int(row["task_index"]) for row in rows}


def _append_jsonl(path: pathlib.Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(row) + "\n")


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    """Writes through a temporary file, so a crash leaves the old file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(partial, path)


def _write_lines(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    """Rewrites a JSON-lines file through a temporary file."""
    partial = path.with_suffix(".jsonl.partial")
    partial.write_text("".join(json.dumps(row) + "\n" for row in rows))
    os.replace(partial, path)


def _vector_stats(values: npt.NDArray[Any]) -> dict[str, list[float]]:
    """Per-dimension statistics of a `[frames, width]` column."""
    values = values.astype(np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def _scalar_stats(values: npt.NDArray[Any]) -> dict[str, list[float]]:
    """Statistics of a `[frames]` column, shaped `[1]` as LeRobot keeps them."""
    return _vector_stats(np.asarray(values).reshape(-1, 1))
