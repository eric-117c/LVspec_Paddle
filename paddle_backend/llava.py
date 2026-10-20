"""LLaVA-OneVision video frontend with a native Paddle SigLIP encoder."""

import math
from types import SimpleNamespace

import paddle
from paddle import nn
from paddle.nn import functional as F

from .cache import ModelCache
from .models import CausalLM, namespace, text_config
from .ops import activation, attention, device_of, move, on_device


class SiglipEmbeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = nn.Conv2D(config.num_channels, config.hidden_size, config.patch_size, stride=config.patch_size, padding=0)
        self.position_embedding = nn.Embedding((config.image_size // config.patch_size)**2, config.hidden_size)

    def forward(self, pixels):
        x = self.patch_embedding(pixels.astype(self.patch_embedding.weight.dtype))
        x = x.flatten(2).transpose([0, 2, 1])
        with paddle.device.device_guard(device_of(x)):
            return x + self.position_embedding(paddle.arange(x.shape[1], dtype="int64"))


class SiglipAttention(nn.Layer):
    def __init__(self, config, mode="sdpa"):
        super().__init__()
        self.mode = mode
        self.heads = config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = [layer(x).reshape([batch, length, self.heads, -1]).transpose([0, 2, 1, 3])
                   for layer in (self.q_proj, self.k_proj, self.v_proj)]
        y = attention(q, k, v, mode=self.mode)
        return self.out_proj(y.transpose([0, 2, 1, 3]).reshape([batch, length, width]))


class SiglipMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = config.hidden_act

    def forward(self, x):
        return self.fc2(activation(self.act, self.fc1(x)))


class SiglipLayer(nn.Layer):
    def __init__(self, config, mode="sdpa"):
        super().__init__()
        self.self_attn = SiglipAttention(config, mode)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class SiglipEncoder(nn.Layer):
    def __init__(self, config, mode="sdpa"):
        super().__init__()
        self.layers = nn.LayerList([SiglipLayer(config, mode) for _ in range(config.num_hidden_layers)])


class SiglipTransformer(nn.Layer):
    def __init__(self, config, mode="sdpa"):
        super().__init__()
        self.embeddings = SiglipEmbeddings(config)
        self.encoder = SiglipEncoder(config, mode)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    @on_device("embeddings.patch_embedding.weight")
    def forward(self, pixels, selected_layer):
        x = self.embeddings(move(pixels, self.embeddings.patch_embedding.weight))
        # HF hidden_states contains embeddings followed by each encoder layer,
        # before post_layernorm (including when selected_layer == -1).
        count = len(self.encoder.layers) + 1
        selected = selected_layer if selected_layer >= 0 else count + selected_layer
        if not 0 <= selected < count:
            raise ValueError("vision_feature_layer is outside the SigLIP hidden state tuple")
        if selected == 0:
            return x
        for i, layer in enumerate(self.encoder.layers, 1):
            x = move(x, layer.layer_norm1.weight)
            with paddle.device.device_guard(device_of(x)):
                x = layer(x)
            if i == selected:
                return x


class SiglipModel(nn.Layer):
    def __init__(self, config, mode="sdpa"):
        super().__init__()
        self.vision_model = SiglipTransformer(config, mode)

    def forward(self, pixels, selected_layer=-1):
        return self.vision_model(pixels, selected_layer)


class Projector(nn.Layer):
    def __init__(self, config):
        super().__init__()
        bias = config.get("multimodal_projector_bias", True)
        self.linear_1 = nn.Linear(config["vision_config"]["hidden_size"], config["text_config"]["hidden_size"], bias_attr=bias)
        self.linear_2 = nn.Linear(config["text_config"]["hidden_size"], config["text_config"]["hidden_size"], bias_attr=bias)
        self.act = config.get("projector_hidden_act", "gelu")

    @on_device("linear_1.weight")
    def forward(self, x):
        return self.linear_2(activation(self.act, self.linear_1(x)))


class LlavaOnevisionForConditionalGeneration(nn.Layer):
    model_type = "llava_ov"

    def __init__(self, config, attention_mode="sdpa"):
        super().__init__()
        if config["vision_config"].get("model_type", "siglip_vision_model") != "siglip_vision_model":
            raise ValueError("LLaVA-OneVision requires a SigLIP vision tower")
        if config["text_config"].get("model_type", "qwen2") != "qwen2":
            raise ValueError("LLaVA-OneVision requires a Qwen2 language model")
        self.raw_config = config
        self.config = text_config(config["text_config"])
        vision = namespace(config["vision_config"], num_channels=3, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6)
        self.vision_config = vision
        self.vision_tower = SiglipModel(vision, "eager" if attention_mode == "eager" else "sdpa")
        self.multi_modal_projector = Projector(config)
        self.image_newline = self.create_parameter([self.config.hidden_size])
        self.language_model = CausalLM(self.config, attention_mode)
        self.video_token_id = config.get("video_token_index", 151647)

    @property
    def text_model(self):
        return self.language_model.model

    def new_cache(self, capacity):
        return ModelCache(self.config.num_hidden_layers, capacity)

    def get_video_features(self, pixels):
        batch, frames, channels, height, width = pixels.shape
        if height != self.vision_config.image_size or width != self.vision_config.image_size:
            raise ValueError("SigLIP video frames must match the configured image_size")
        features = self.vision_tower(pixels.reshape([batch * frames, channels, height, width]), self.raw_config.get("vision_feature_layer", -1))
        if self.raw_config.get("vision_feature_select_strategy", "full") != "full":
            raise ValueError("SigLIP video inference requires vision_feature_select_strategy='full'")
        features = self.multi_modal_projector(move(features, self.multi_modal_projector.linear_1.weight))
        side = self.vision_config.image_size // self.vision_config.patch_size
        with paddle.device.device_guard(device_of(features)):
            features = features.reshape([batch * frames, side, side, -1]).transpose([0, 3, 1, 2])
            features = F.interpolate(features, size=[math.ceil(side / 2)] * 2, mode="bilinear", align_corners=False, align_mode=0)
            features = features.transpose([0, 2, 3, 1]).reshape([batch, -1, self.config.hidden_size])
            newline = move(self.image_newline, features).reshape([1, 1, -1]).expand([batch, 1, -1])
            return paddle.concat([features, newline], axis=1).reshape([-1, self.config.hidden_size])

    @on_device("language_model.model.embed_tokens.weight")
    def forward(self, input_ids, cache=None, pixel_values_videos=None, selected_indices=None,
                position_ids=None, tree_mask=None, attention_mask=None, logits_to_keep=0,
                pixel_values=None, **kwargs):
        if pixel_values is not None:
            raise ValueError("This LVSpec backend implements LLaVA video inference; image AnyRes inputs are unsupported")
        if kwargs.keys() - {"image_sizes_videos", "image_sizes"}:
            raise TypeError(f"Unsupported model inputs: {sorted(kwargs)}")
        embed = self.text_model.embed_tokens
        ids = move(input_ids, embed.weight)
        x = embed(ids)
        video = None
        if pixel_values_videos is not None:
            video = self.get_video_features(pixel_values_videos)
            if selected_indices is not None:
                video = paddle.index_select(video, move(selected_indices, video))
            video = move(video, x).astype(x.dtype)
            locations = paddle.nonzero(ids[0] == self.video_token_id).flatten()
            if len(locations) != video.shape[0]:
                raise ValueError(f"Visual token/feature mismatch: {len(locations)} != {video.shape[0]}")
            x[0, locations] = video
        hidden = self.text_model(inputs_embeds=x, cache=cache, position_ids=position_ids, tree_mask=tree_mask, attention_mask=attention_mask)
        head = self.language_model.lm_head
        with paddle.device.device_guard(device_of(head.weight)):
            logits = head(move(hidden[:, -logits_to_keep:] if logits_to_keep else hidden, head.weight))
        return SimpleNamespace(logits=logits, hidden_states=hidden, video_hidden_states=video, output_embeddings=x)
