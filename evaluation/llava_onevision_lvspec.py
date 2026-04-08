import copy
import json
import logging
import math
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

from PIL import Image

# LVSpec & SpecVLM imports
from lmms_eval.LVSpec.models.modeling_llava_onevision_tree import LlavaOnevisionForConditionalGeneration
from lmms_eval.LVSpec.decoding.tree_decoding import *
from lmms_eval.LVSpec.tree_choices.choices import mc_sim_7b_63, chain, chain_15, chain2_8, chain2_10
from lmms_eval.LVSpec.models.transformers_local.models.llava_onevision.processing_llava_onevision import LlavaOnevisionProcessor

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

# Import LLaVA modules
from lmms_eval.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

from lmms_eval.llava.conversation import SeparatorStyle, conv_templates
from lmms_eval.llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)

# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava_onevision")
class Llava_OneVision_specvlm(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "pyav",
        #new
        method: Optional[str] = "ar",
        gen_length: Optional[int] = 512,
        drop_rate: Optional[float] = 0.9,
        window: Optional[int] = 2,
        loose_random: Optional[float] = 0.5,
        loose_entropy: Optional[float] = 0.1,
        loose_hidden_score: Optional[float] = 0.5,
        loose_hidden_k: Optional[int] = 7,
        similarity_k: Optional[int] = 10,
        loose_rank: Optional[float] = 0.5,
        draft_pretrained: Optional[str] = None,
        result_path: Optional[str] = None,
        tree_choice: Optional[str] = None,
        skip: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        # accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        # if accelerator.num_processes > 1:
        #     self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        #     self.device_map = f"cuda:{accelerator.local_process_index}"
        # elif accelerator.num_processes == 1 and device_map == "auto":
        #     self._device = torch.device(device)
        #     self.device_map = device_map
        # else:
        #     self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        #     self.device_map = f"cuda:{accelerator.local_process_index}"

        # #MODIFIED
        self._device = torch.device(device)
        self.device_map = device_map

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

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
        self.skip=skip

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        llava_model_args["overwrite_config"] = overwrite_config
        try:
            self.processor = LlavaOnevisionProcessor.from_pretrained(pretrained, device_map="auto", torch_dtype=torch.float16)
            self._tokenizer = self.processor.tokenizer
            self._image_processor = self.processor.image_processor
            self._model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                pretrained, 
                device_map="auto", 
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
            # self._model = accelerator.prepare(self._model)
            self._max_length = 49152

            if draft_pretrained is not None:
                # _, self.draft_model, _, _ = load_pretrained_model(draft_pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
                self.draft_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    draft_pretrained, 
                    device_map="auto", 
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                )
            self.video_token_id = 151647
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            # llava_model_args.pop("multimodal", None)
            # self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
            print("Loading model failed")

        self._config = self._model.config
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

        # if accelerator.num_processes > 1:
        #     assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
        #     # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
        #     # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
        #     # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
        #     if accelerator.distributed_type == DistributedType.DEEPSPEED:
        #         kwargs = {
        #             "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
        #             "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
        #         }
        #         AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
        #         eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

        #     if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
        #         self._model = accelerator.prepare(self.model)
        #     else:
        #         self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
        #     self.accelerator = accelerator
        #     if self.accelerator.is_local_main_process:
        #         eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        #     self._rank = self.accelerator.local_process_index
        #     self._world_size = self.accelerator.num_processes

        # elif accelerator.num_processes == 1 and device_map == "auto":
        #     eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
        #     self._rank = 0
        #     self._world_size = 1

        # else:
        #     eval_logger.info(f"Using single device: {self._device}")
        #     self.model.to(self._device)
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
    #         return self.accelerator.unwrap_model(self._model)
    #     else:
    #         return self._model

    #MODIFIED
    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                self._config.image_aspect_ratio = origin_image_aspect_ratio
                eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

            if visual is None or visual == []:
                visual = None
                task_type = "text"
                image_tensor = None
            else:
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:
                    self._config.image_aspect_ratio = "pad"
                    eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                if "task_type" in self.metadata and self.metadata["task_type"] == "video" and "sample_frames" in self.metadata:
                    assert type(visual) == list, "sample_frames must be specified for video task"
                    sample_indices = np.linspace(0, len(visual) - 1, self.metadata["sample_frames"], dtype=int)
                    visual = [visual[i] for i in sample_indices]
                    assert len(visual) == self.metadata["sample_frames"]

                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                    task_type = "video"

                # elif type(visual[0]) == PIL.Image.Image:
                elif isinstance(visual[0], PIL.Image.Image):
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                    task_type = "image"

                elif type(visual[0]) == str:
                    image_tensor = []
                    try:
                        if self.video_decode_backend == "decord":
                            frames = self.load_video(visual, self.max_frames_num)
                        elif self.video_decode_backend == "pyav":
                            frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                        frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                        image_tensor.append(frames)
                    except Exception as e:
                        eval_logger.error(f"Error {e} in loading video")
                        image_tensor = None

                    task_type = "video"

            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in contexts:
                placeholder_count = len(visual) if isinstance(visual, list) else 1
                if task_type == "video":
                    placeholder_count = len(frames) if self.token_strategy == "multiple" else 1
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + contexts
            else:
                prompts_input = contexts

            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])

            conv.messages[-1][1] = continuation
            full_prompt = conv.get_prompt()
            full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            labels = full_input_ids.clone()
            labels[0, : input_ids.shape[1]] = -100

            kwargs = {}
            if task_type == "image":
                kwargs["image_sizes"] = [[v.size[0], v.size[1]] for v in visual] if isinstance(visual, list) else [[visual.size[0], visual.size[1]]]
            elif task_type == "video":
                kwargs["modalities"] = ["video"]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            with torch.inference_mode():
                outputs = self.model(input_ids=full_input_ids, labels=labels, images=image_tensor, use_cache=True, **kwargs)

            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = full_input_ids[:, input_ids.shape[1] :]
            greedy_tokens = greedy_tokens[:, input_ids.shape[1] : full_input_ids.shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()

            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        # MODIFIED
        decoding_times = []
        accept_lengths = []
        generate_lens = []

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            print("Task:", task)
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            # import ipdb; ipdb.set_trace()
            for visual, context in zip(batched_visuals, batched_contexts):
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                if visual is None or visual == []:  # for text-only tasks.
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                else:
                    if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                        self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                        eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                    if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                        assert type(visual) == list, "sample_frames must be specified for video task"
                        sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                        visual = [visual[i] for i in sample_indices]
                        assert len(visual) == metadata["sample_frames"]

                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                        task_type = "video"
                        placeholder_count = 1
                        # print("multi image task")

                    elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                        task_type = "image"
                        placeholder_count = len(visual) if isinstance(visual, list) else 1
                        # print("image task")

                    elif type(visual[0]) == str:  # For video task
                        # print("video task")
                        image_tensor = []

                        if self.video_decode_backend == "decord":
                            print("Loading video with decord backend...")
                            frames = self.load_video(visual, self.max_frames_num)
                            print("frames shape:", frames.shape)
                        elif self.video_decode_backend == "pyav":
                            # frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)

                            # MODIFIED
                            image_tensor = None
                            container = av.open(visual[0])
                            total_frames = container.streams.video[0].frames

                            if total_frames == 0:
                                video = None
                            else:
                                indices = np.arange(0, total_frames, total_frames / self.max_frames_num).astype(int)
                                video = read_video_pyav(container, indices)
                            

                        # frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                        # image_tensor.append(frames)

                        task_type = "video"
                        placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                    """
                    # if task_type == "image": # indeed in multi-image case, not the video in frames.
                    #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    # elif task_type == "video":
                    # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                # MODIFIED
                if task_type == "video":
                    video_tokens = ["<video>"] * placeholder_count
                    video_tokens = " ".join(video_tokens)
                    question = video_tokens + "\n" + context

                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):  # conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # only simple string for question
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            #MODIFIED
            # print("video shape:", video.shape)
            # print(question_input[0])
            if task_type == "video":
                if video is not None:
                    inputs = self.processor(videos=list(video), text=question_input[0], return_tensors="pt").to("cuda")
                else:
                    inputs = self.processor(videos=None, text=question_input[0], return_tensors="pt").to("cuda")
            else:
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
                attention_masks = input_ids.ne(pad_token_ids).to(self.device)

                # MODIFIED
                inputs = {}
                inputs['input_ids'] = input_ids
                inputs['attention_mask'] = attention_masks
                print(image_tensor.shape)
                inputs['pixel_values'] = image_tensor

            if task_type == "image":
                gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            elif task_type == "video":
                # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                # keywords = [stop_str]
                # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"]
                # gen_kwargs["stopping_criteria"] = [stopping_criteria]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024

            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")
            # When do_sample=False, remove sampling-related parameters to avoid warnings
            # These might be in gen_kwargs or in the model's generation_config
            if not gen_kwargs.get("do_sample", False):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                gen_kwargs.pop("top_k", None)
            try:
                with torch.inference_mode():
                    # cont = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)
                    
                    input_ids_copy = inputs['input_ids'].clone()
                    print("Inputs_shape:", inputs['input_ids'].shape[1])
                    if self.method == "ar":
                        output = AR_generate_one_stage(
                            inputs, 
                            self.model,
                            max_new_tokens=self.gen_length,
                            video_token_id=151647, 
                            processor=self.processor
                            )
                    elif self.method == "sd":
                        output = SD_generate(
                            inputs,
                            self.model,
                            self.draft_model,
                            self.processor,
                            max_new_tokens=self.gen_length,
                            video_token_id=151647,
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
                            video_token_id = 151647,
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
                            video_token_id = 151647,
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




                    # text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                    if task == "videomme" or task == "videomme_long":
                        text_outputs = self.tokenizer.batch_decode(output['output_ids'][:, input_ids_copy.shape[1]:], skip_special_tokens=True)
                        # print("Trimmed generated text outputs:", text_outputs)
                    else:
                        text_outputs = self.tokenizer.batch_decode(output['output_ids'], skip_special_tokens=True)
                        # print("Generated text outputs:", text_outputs)
                        # print(output['output_ids'].shape)

            except Exception as e:
                raise e

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)



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

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_to_text, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            # multi round inference: terminate when receiving signal from the doc_to_text
            round_idx = 0
            batched_round_res = []
            batched_previous_round_info = None
            while True:
                question_input = []

                if round_idx != 0:  # get current round visual and context from doc_to_text function
                    batched_visuals, batched_contexts, batched_terminal_singal, batched_round_res, batched_previous_round_info = list(
                        zip(
                            *[
                                batched_doc_to_text[0](
                                    self.task_dict[task][split][ids],
                                    previous_output=[round_res[ids_idx] for round_res in batched_round_res],
                                    round_idx=round_idx,
                                    previous_round_info=batched_previous_round_info[ids_idx] if batched_previous_round_info is not None else None,
                                )
                                for ids_idx, ids in enumerate(batched_doc_id)
                            ]
                        )
                    )
                    # import ipdb; ipdb.set_trace()
                    batched_round_res = list(zip(*batched_round_res))  # [(r1_1, r1_2), (r2_1, r2_2), ...]
                    if batched_terminal_singal[0]:  # terminal signal from doc_to_text function
                        break

                for visual, context in zip(batched_visuals, batched_contexts):
                    if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                        self._config.image_aspect_ratio = origin_image_aspect_ratio
                        eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                    if visual is None or visual == []:  # for text-only tasks.
                        visual = None
                        task_type = "text"
                        placeholder_count = 0
                        image_tensor = None
                    else:
                        if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                            self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                            eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                        if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                            assert type(visual) == list, "sample_frames must be specified for video task"
                            sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                            visual = [visual[i] for i in sample_indices]
                            assert len(visual) == metadata["sample_frames"]

                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                            task_type = "video"
                            placeholder_count = 1

                        elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                            task_type = "image"
                            placeholder_count = len(visual) if isinstance(visual, list) else 1

                        elif type(visual[0]) == str:  # For video task
                            image_tensor = []
                            try:
                                if self.video_decode_backend == "decord":
                                    frames = self.load_video(visual, self.max_frames_num)
                                elif self.video_decode_backend == "pyav":
                                    frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                                frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                                image_tensor.append(frames)
                            except Exception as e:
                                eval_logger.error(f"Error {e} in loading video")
                                image_tensor = None

                            task_type = "video"
                            placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                    if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                        """
                        Three senarios:
                        1. No image, and there for, no image token should be added.
                        2. image token is already specified in the context, so we don't need to add it.
                        3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                        4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                        """
                        # if task_type == "image": # indeed in multi-image case, not the video in frames.
                        #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                        # elif task_type == "video":
                        # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                        image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                        image_tokens = " ".join(image_tokens)
                        question = image_tokens + "\n" + context
                    else:
                        question = context

                    # This is much safer for llama3, as we now have some object type in it
                    if "llama_3" in self.conv_template:
                        conv = copy.deepcopy(conv_templates[self.conv_template])
                    else:
                        conv = conv_templates[self.conv_template].copy()

                    if utils.is_json(question):  # conversational question input
                        question = json.loads(question)
                        for idx, item in enumerate(question):
                            role = conv.roles[idx % 2]
                            message = item["value"]
                            conv.append_message(role, message)

                        assert len(conv.messages) % 2 == 1
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)
                    else:  # only simple string for question
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)

                # preconfigure gen_kwargs with defaults
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = False
                # Only set temperature and top_p if do_sample is True
                if gen_kwargs.get("do_sample", False):
                    if "temperature" not in gen_kwargs:
                        gen_kwargs["temperature"] = 1.0  # Default temperature for sampling
                    if "top_p" not in gen_kwargs:
                        gen_kwargs["top_p"] = 1.0  # Default top_p for sampling
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
                print("input_ids_list length:", input_ids_list[0])
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
                attention_masks = input_ids.ne(pad_token_ids).to(self.device)

                if task_type == "image":
                    gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
                elif task_type == "video":
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                    gen_kwargs["modalities"] = ["video"]
                    gen_kwargs["stopping_criteria"] = [stopping_criteria]
                    self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                    self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

                # These steps are not in LLaVA's original code, but are necessary for generation to work
                # TODO: attention to this major generation step...
                if "image_aspect_ratio" in gen_kwargs.keys():
                    gen_kwargs.pop("image_aspect_ratio")
                # Remove temperature and top_p when do_sample=False to avoid warnings
                if not gen_kwargs.get("do_sample", False):
                    gen_kwargs.pop("temperature", None)
                    gen_kwargs.pop("top_p", None)
                try:
                    with torch.inference_mode():
                        cont = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)
                        # cont = self.model.generate(qwen_input_ids, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)

                    text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                except Exception as e:
                    raise e

                text_outputs = [response.strip() for response in text_outputs]
                batched_round_res.append(text_outputs)

                round_idx += 1

            res.extend(list(zip(*batched_round_res)))
            self.cache_hook.add_partial("generate_until_multi_round", (context, gen_kwargs), batched_round_res)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
