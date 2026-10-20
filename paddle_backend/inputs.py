"""NumPy/Hugging Face preprocessing; model execution stays in Paddle."""

import json
from pathlib import Path

import av
import numpy as np
import paddle


def load_processor(path, model_type, max_pixels=448 * 448):
    if model_type == "qwen2_5_vl":
        from models.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        # In Transformers 4.48, passing max_pixels to from_pretrained also
        # forwards it to preprocess(), whose signature does not accept it.
        processor = Qwen2_5_VLProcessor.from_pretrained(path)
        if max_pixels < processor.image_processor.min_pixels:
            raise ValueError("max_pixels must be at least the processor's min_pixels")
        processor.image_processor.max_pixels = max_pixels
        if isinstance(getattr(processor.image_processor, "size", None), dict):
            processor.image_processor.size["max_pixels"] = max_pixels
        return processor
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(path)


def read_video(path, frame_num):
    if frame_num < 1:
        raise ValueError("frame_num must be positive")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        count = stream.frames
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        if not count:
            # Some valid containers do not advertise a frame count.
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            if not frames:
                raise ValueError(f"No decodable frames: {path}")
            idx = np.linspace(0, len(frames) - 1, frame_num).astype(np.int64)
            return np.stack([frames[i] for i in idx]), frame_num / duration if duration else 2.0
        idx = np.linspace(0, count - 1, frame_num).astype(np.int64)
        selected, wanted = {}, set(idx.tolist())
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                selected[i] = frame.to_ndarray(format="rgb24")
            if i >= idx[-1]:
                break
        if set(selected) != wanted:
            raise ValueError(f"Video metadata reports frames that cannot be decoded: {path}")
        return np.stack([selected[i] for i in idx]), frame_num / duration if duration else 2.0


def prepare_frames(processor, frames, prompt, model_type, fps=2.0):
    conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    kwargs = {"fps": fps} if model_type == "qwen2_5_vl" else {}
    batch = processor(text=[text], videos=[frames], return_tensors="np", **kwargs)
    return {key: np.asarray(value) for key, value in batch.items()}


def to_paddle(inputs, device="cpu"):
    result = {}
    for key, value in inputs.items():
        if value is None:
            result[key] = None
        elif isinstance(value, paddle.Tensor):
            result[key] = value.to(device)
        else:
            array = np.asarray(value)
            if array.dtype.kind in "iu":
                array = array.astype(np.int64)
            elif array.dtype.kind == "f":
                array = array.astype(np.float32)
            result[key] = paddle.to_tensor(array, place=device)
    return result


def read_samples(data_path, task="VideoDetailCaption", data_num=100):
    path = Path(data_path)
    if path.suffix in (".json", ".jsonl"):
        text = path.read_text()
        samples = json.loads(text) if path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
        if not isinstance(samples, list):
            raise ValueError("Sample manifest must be a list of {video, question} objects")
        for sample in samples[:data_num]:
            video = Path(sample["video"])
            yield {**sample, "video_path": str(video if video.is_absolute() else path.parent / video)}
        return
    from datasets import load_dataset
    dataset = load_dataset(str(path), split="test").shuffle(seed=42)
    for sample in dataset.select(range(min(data_num, len(dataset)))):
        if task == "VideoDetailCaption":
            video = path / "Test_Videos" / f"{sample['video_name']}.mp4"
        else:
            video = path / sample.get("video_path", sample.get("video", ""))
        if video.is_file():
            yield {**sample, "video_path": str(video), "question": sample.get("question", "Describe the video in detail.")}
