import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    # Qwen2_5_VLForConditionalGeneration,
)

from lmms_eval.SpecVLM.models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
# from lmms_eval.SpecVLM.models.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from lmms_eval.SpecVLM.decoding.tree_decoding_qwen2_5 import *
from lmms_eval.SpecVLM.tree_choices.choices import mc_sim_7b_63, chain, chain_15, chain2_8, chain2_10
# from lmms_eval.SpecVLM.utils.utils import drop_visual_tokens_uniform

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl")
class Qwen2_5_VL_specvlm(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        #new
        method: Optional[str] = "ar",
        gen_length: Optional[int] = 512,
        drop_rate: Optional[float] = 0.9,
        window: Optional[int] = 2,
        loose_random: Optional[float] = 0.5,
        loose_entropy: Optional[float] = 0.1,
        loose_hidden_score: Optional[float] = 0.5,
        loose_hidden_k: Optional[int] = 3,
        similarity_k: Optional[int] = 10,
        loose_rank: Optional[float] = 0.5,
        #
        draft_pretrained: Optional[str] = None,
        result_path: Optional[str] = None,
        tree_choice: Optional[str] = None,
        skip: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        # accelerator = Accelerator()
        # self.accelerator = accelerator
        # if accelerator.num_processes > 1:
        #     self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        #     self.device_map = f"cuda:{accelerator.local_process_index}"
        # else:
        #     self._device = torch.device(device)
        #     self.device_map = device_map if device_map else device

        #MODIFIED
        self._device = torch.device(device)
        self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()

        # MODIFIED
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained, 
            device_map="auto", 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            attn_implementation = "sdpa",
        ).eval()

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        #new
        self.method = method #MODIFIED
        self.gen_length = gen_length
        self.drop_rate = drop_rate
        self.loose_window = window
        self.loose_random = loose_random
        self.loose_entropy = loose_entropy
        self.loose_hidden_score = loose_hidden_score
        self.loose_hidden_k = loose_hidden_k
        self.similarity_k = similarity_k
        self.loose_rank = loose_rank
        if tree_choice == "mc_sim_7b_63":
            self.tree_choices = mc_sim_7b_63
        elif tree_choice == "chain":
            self.tree_choices = chain
        elif tree_choice == 'chain_15':
            self.tree_choices = chain_15
        elif tree_choice == 'chain2_8':
            self.tree_choices = chain2_8
        elif tree_choice == 'chain2_10':
            self.tree_choices = chain2_10
        else:
            self.tree_choices = chain
        self.result_path = result_path
        self.skip = skip
        if draft_pretrained is not None:
            self.draft_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                draft_pretrained, 
                device_map="auto", 
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                attn_implementation = "sdpa",
            ).eval()

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        # if accelerator.num_processes > 1:
        #     print("-------------Preparing model with accelerator...")
        #     assert accelerator.distributed_type in [
        #         DistributedType.FSDP,
        #         DistributedType.MULTI_GPU,
        #     ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
        #     if accelerator.distributed_type == DistributedType.FSDP:
        #         self._model = accelerator.prepare(self.model)
        #     else:
        #         self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
        #     self.accelerator = accelerator
        #     if self.accelerator.is_local_main_process:
        #         eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        #     self._rank = self.accelerator.local_process_index
        #     self._world_size = self.accelerator.num_processes
        # else:
        #     self._rank = 0
        #     self._world_size = 1

        #MODIFIED
        self._rank = 0
        self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    # @property
    # def model(self):
    #     # returns the model, unwrapping it if using Accelerate
    #     if hasattr(self, "accelerator"):
    #         print("---------------Unwrapping model from accelerator...")
    #         return self.accelerator.unwrap_model(self._model)
    #     else:
    #         return self._model

    # @property
    # def model(self):
    #     return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        # MODIFIED
        decoding_times = []
        accept_lengths = []
        generate_lens = []

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            print("--------Task:", task)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
                            # max_pixels = height * width
                            processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            base64_image = visual.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Ensure unique indices if linspace produces duplicates for few frames
                indices = np.unique(indices)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)  # Ensure uniqueness again
                video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

            #MODIFIED
            if self.skip:
                # Otherwise OOM
                print("Raw input length:", inputs.input_ids.shape[1])
                if inputs.input_ids.shape[1] > self.max_num_frames * 200:
                    print("skip:large")
                    modified_frames = int(self.max_num_frames / 2)
                # elif inputs.input_ids.shape[1] < self.max_num_frames * 50:
                #     print("skip:small")
                #     modified_frames = int(self.max_num_frames * 2)
                    image_inputs, video_inputs = process_vision_info(batched_messages)
                    if video_inputs is not None:
                        total_frames = video_inputs[0].shape[0]
                        indices = np.linspace(0, total_frames - 1, modified_frames, dtype=int)
                        # Ensure unique indices if linspace produces duplicates for few frames
                        indices = np.unique(indices)
                        # Append the last frame index if not already included
                        if total_frames - 1 not in indices:
                            indices = np.append(indices, total_frames - 1)
                            indices = np.unique(indices)  # Ensure uniqueness again
                        video_inputs[0] = video_inputs[0][indices]
                    inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")


            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 32768,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            # cont = self.model.generate(
            #     **inputs,
            #     eos_token_id=self.tokenizer.eos_token_id,
            #     pad_token_id=pad_token_id,
            #     do_sample=current_gen_kwargs["do_sample"],
            #     temperature=current_gen_kwargs["temperature"],
            #     top_p=current_gen_kwargs["top_p"],
            #     num_beams=current_gen_kwargs["num_beams"],
            #     max_new_tokens=current_gen_kwargs["max_new_tokens"],
            #     use_cache=self.use_cache,
            # )
            
            print("Input length:", inputs.input_ids.shape[1])
            #MODIFIED
            # if self.skip:
            #     # Otherwise OOM
            #     if inputs.input_ids.shape[1] > self.max_num_frames * 220 or inputs.input_ids.shape[1] < self.max_num_frames * 100:
            #         print("skip")
            #         # res.append("Ignore this sample and do not count towards evaluation. Let and score and acc be -1.")
            #         res.append(None)
            #         pbar.update(1)
            #         continue

            input_ids_copy = inputs.input_ids.clone()
            if self.method == "ar":
                output = AR_generate_one_stage(
                    inputs, 
                    self.model,
                    max_new_tokens=self.gen_length,
                    video_token_id=151656, 
                    processor=self.processor
                    )
            elif self.method == "sd":
                output = SD_generate(
                    inputs,
                    self.model,
                    self.draft_model,
                    self.processor,
                    max_new_tokens=self.gen_length,
                    video_token_id=151656,
                    tree_choices=self.tree_choices,
                    )
            elif self.method == "specvlm":
                output = SD_generate_with_pruning(
                    inputs,
                    self.model,
                    self.draft_model,
                    self.processor,
                    method = 'uniform',
                    drop_rate = self.drop_rate,
                    video_token_id = 151656,
                    max_new_tokens=self.gen_length,
                    tree_choices=self.tree_choices,
                    percentage=0.4,
                    )
            elif self.method == "lvspec":
                output = SD_generate_with_pruning(
                    inputs,
                    self.model,
                    self.draft_model,
                    self.processor,
                    method = 'uniform',
                    drop_rate = self.drop_rate,
                    video_token_id = 151656,
                    max_new_tokens=self.gen_length,
                    tree_choices=self.tree_choices,
                    percentage=0.4,
                    loose='hidden_topk_or_search',
                    loose_hidden_k=self.loose_hidden_k,
                    similarity_k=self.similarity_k,
                    )
            else:
                raise ValueError(f"Unknown generation method: {self.method}")

            # MODIFIED
            decoding_times.append(output['decoding_time'])
            generate_lens.append(output['generate_len'])
            if self.method == "ar" or self.method == "ar_random":
                accept_lengths.append(0)
            else:
                output['mean_accept_length'] = float(output['mean_accept_length'])
                accept_lengths.append(output['mean_accept_length'])

            
            # generated_ids_trimmed = output_ar['output_ids'][:, inputs.input_ids.shape[1]:]
            # answers = self.processor.batch_decode(output_ar['output_ids'], skip_special_tokens=True)[0]
            
            print("Generated_ids_shape:", output['output_ids'].shape)
            # print("Input_ids_shape:", inputs.input_ids.shape)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids_copy, output['output_ids'])]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # print(answers)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                # eval_logger.debug(f"Question: {context}")
                # eval_logger.debug(f"Model Raw Response: {ans}")
                # eval_logger.debug(f"Model Clean Response: {clean_ans}")

        # MODIFIED
        # Save average metirc to a json file in self.result_path
        if self.result_path is not None and self.rank == 0:
            avg_decoding_time = float(sum(decoding_times) / len(decoding_times))
            avg_accept_length = float(sum(accept_lengths) / len(accept_lengths))
            avg_generate_len = float(sum(generate_lens) / len(generate_lens))
            avg_tokens_per_second = avg_generate_len / avg_decoding_time if avg_decoding_time > 0 else 0.0
            result_dict = {
                "decoding_time": avg_decoding_time,
                "average_accept_length": avg_accept_length,
                "average_generate_length": avg_generate_len,
                "tokens_per_second": avg_tokens_per_second,
            }
            print(result_dict)
            import json
            import os
            # from datetime import datetime
            # current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_json_path = os.path.join(
                self.result_path, "metric.json"
            )
            os.makedirs(self.result_path, exist_ok=True)
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=4)
            eval_logger.info(f"Saved generation metrics to {result_json_path}")
        
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
