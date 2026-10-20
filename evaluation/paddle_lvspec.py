"""lmms-eval adapter. Register as paddle_lvspec in lmms_eval/models/__init__.py.

Run with one process; the model shards its layers over the requested logical GPUs.
The adapter handles video tasks whose doc_to_visual returns one local video path.
"""

import json
from pathlib import Path

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("paddle_lvspec")
class PaddleLVSpec(lmms):
    def __init__(self, pretrained, draft_pretrained=None, model_type="qwen2_5_vl",
                 method="lvspec", batch_size=1, dtype="float16", devices="0",
                 draft_devices=None, frame_num=64, max_pixels=448 * 448,
                 max_new_tokens=512, drop_rate=0.9, loose_hidden_k=7,
                 attention="sdpa", result_path=None, **kwargs):
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected model arguments: {sorted(kwargs)}")
        if int(batch_size) != 1:
            raise ValueError("Paddle LVSpec supports batch_size=1")
        if method != "ar" and draft_pretrained is None:
            raise ValueError("draft_pretrained is required for speculative decoding")
        import paddle
        from paddle_backend.checkpoint import load_model, load_model_pair, resolve_path
        from paddle_backend.inputs import load_processor
        if paddle.distributed.get_world_size() != 1:
            raise ValueError("Use one lmms-eval process; this backend performs its own layer sharding")
        self.devices = ["cpu"] if devices == "cpu" else [f"gpu:{int(i)}" for i in str(devices).split(":")]
        draft_places = self.devices[-1:] if draft_devices is None else (["cpu"] if draft_devices == "cpu" else [f"gpu:{int(i)}" for i in str(draft_devices).split(":")])
        if method != "ar":
            self._model, self.draft_model = load_model_pair(pretrained, draft_pretrained, model_type, dtype,
                                                           self.devices, draft_places, attention)
        else:
            self._model = load_model(pretrained, model_type, dtype, self.devices, attention)
            self.draft_model = None
        self.processor = load_processor(str(resolve_path(pretrained)), model_type, int(max_pixels))
        if self.draft_model is not None:
            draft_processor = load_processor(str(resolve_path(draft_pretrained)), model_type, int(max_pixels))
            if self.processor.tokenizer.get_vocab() != draft_processor.tokenizer.get_vocab():
                raise ValueError("Target and draft tokenizer vocabularies differ")
        self.model_type, self.method = model_type, method
        self.frame_num, self.max_new_tokens = int(frame_num), int(max_new_tokens)
        self.drop_rate, self.loose_hidden_k = float(drop_rate), int(loose_hidden_k)
        self.result_path = Path(result_path) if result_path else None
        self._rank, self._world_size = 0, 1

    @property
    def config(self):
        return self._model.config

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def model(self):
        return self._model

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self.devices[0]

    @property
    def max_length(self):
        return self._model.config.max_position_embeddings

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def loglikelihood(self, requests):
        raise NotImplementedError("Paddle LVSpec implements video generation evaluation")

    def generate_until(self, requests):
        from paddle_backend.decoding import generate
        from paddle_backend.inputs import prepare_frames, read_video, to_paddle
        answers = []
        for index, request in enumerate(requests):
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            if gen_kwargs.get("do_sample", False) or float(gen_kwargs.get("temperature") or 0) > 0 or int(gen_kwargs.get("num_beams", 1)) != 1:
                raise ValueError("LVSpec currently implements greedy generation only")
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if isinstance(visuals, (str, Path)):
                visuals = [visuals]
            if not visuals or len(visuals) != 1 or not isinstance(visuals[0], (str, Path)):
                raise ValueError("doc_to_visual must return exactly one local video path")
            frames, fps = read_video(visuals[0], self.frame_num)
            inputs = prepare_frames(self.processor, frames, context.replace("<image>", "").replace("<video>", ""), self.model_type, fps)
            inputs = to_paddle(inputs, self.devices[0])
            output = generate(inputs, self._model, self.draft_model, self.method,
                              max_new_tokens=int(gen_kwargs.get("max_new_tokens", self.max_new_tokens)),
                              drop_rate=self.drop_rate, loose_hidden_k=self.loose_hidden_k,
                              eos_token_id=self.eot_token_id)
            suffix = output["output_ids"][:, inputs["input_ids"].shape[-1]:].numpy()
            answer = self.processor.batch_decode(suffix, skip_special_tokens=True)[0]
            stops = gen_kwargs.get("until", [])
            stops = [stops] if isinstance(stops, str) else stops
            for stop in stops:
                if stop:
                    answer = answer.split(stop, 1)[0]
            answers.append(answer)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
            if self.result_path:
                self.result_path.parent.mkdir(parents=True, exist_ok=True)
                row = dict(sample_index=index, task=task, doc_id=str(doc_id), method=self.method,
                           backend="paddle", decoding_time=output["decoding_time"],
                           generate_len=output["generate_len"], average_accept_length=output["mean_accept_length"], output=answer)
                with self.result_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
        return answers
