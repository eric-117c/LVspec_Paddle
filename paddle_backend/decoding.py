"""Tree speculative decoding, shared across Qwen and LLaVA Paddle models."""

import time

import numpy as np
import paddle
from paddle.nn import functional as F

from .ops import device_of, move, on_device


def synchronize(models):
    devices = set()
    for model in models:
        if model is not None:
            devices.update(p.place.gpu_device_id() for p in model.parameters() if p.place.is_gpu_place())
    for device in sorted(devices):
        paddle.device.synchronize(f"gpu:{device}")


def generate_tree_buffers(choices):
    paths = [()] + sorted({tuple(path) for path in choices}, key=lambda p: (len(p), p))
    if len(paths) == 1:
        raise ValueError("The speculative tree must contain at least one edge")
    lookup = {path: i for i, path in enumerate(paths)}
    for path in paths[1:]:
        if path[:-1] not in lookup or any(not isinstance(rank, int) or rank < 0 for rank in path):
            raise ValueError("Tree choices must contain every ancestor and nonnegative integer ranks")
    allowed = np.zeros((len(paths), len(paths)), dtype=bool)
    for i, path in enumerate(paths):
        for length in range(len(path) + 1):
            allowed[i, lookup[path[:length]]] = True
    parents = {path[:-1] for path in paths[1:]}
    leaves = sorted(path for path in paths if path not in parents)
    retrieve = [[lookup[path[:depth]] for depth in range(len(path) + 1)] for path in leaves]
    return {"paths": paths, "lookup": lookup, "mask": allowed,
            "positions": [len(p) for p in paths], "retrieve_indices": retrieve,
            "greedy_chain": all(path == (0,) * i for i, path in enumerate(paths))}


@on_device("input_ids", keyword="inputs")
def drop_visual_tokens_uniform(inputs, drop_rate=0.9, visual_token_id=151656):
    if not 0 <= drop_rate <= 1:
        raise ValueError("drop_rate must be between 0 and 1")
    ids = inputs["input_ids"]
    if ids.shape[0] != 1:
        raise ValueError("Visual pruning currently supports batch_size=1")
    positions = np.flatnonzero(ids[0].numpy() == visual_token_id)
    count = int(len(positions) * (1 - drop_rate))
    # Match torch.linspace(..., dtype=long), which truncates towards zero.
    selected = np.linspace(0, len(positions) - 1, count).astype(np.int64) if count else np.empty(0, dtype=np.int64)
    keep = ids[0].numpy() != visual_token_id
    keep[positions[selected]] = True
    result = dict(inputs)
    indices = paddle.to_tensor(np.flatnonzero(keep), dtype="int64", place=ids.place)
    result["input_ids"] = paddle.index_select(ids, indices, axis=1)
    result["selected_indices"] = paddle.to_tensor(selected, dtype="int64", place=ids.place)
    result["attention_mask"] = None
    return result


def _normalized_visual_chunks(prefill_hidden, tree_hidden, chunk_size=2048):
    visual = prefill_hidden[0] if prefill_hidden.ndim == 3 else prefill_hidden
    if visual.shape[0] == 0:
        raise ValueError("Visual similarity requires nonempty prefill hidden states")
    visual = move(visual, tree_hidden).astype("float32")
    return tuple(F.normalize(visual[start:start + chunk_size], axis=-1)
                 for start in range(0, visual.shape[0], chunk_size))


def visual_similarity_hidden_state(prefill_hidden, tree_hidden, topk=10, chunk_size=2048,
                                  normalized_reference=None):
    parts = (_normalized_visual_chunks(prefill_hidden, tree_hidden, chunk_size)
             if normalized_reference is None else normalized_reference)
    tree = F.normalize(tree_hidden.astype("float32"), axis=-1)
    k = min(topk, sum(part.shape[0] for part in parts))
    best = None
    for part in parts:
        scores = paddle.matmul(tree, part, transpose_y=True)
        combined = scores if best is None else paddle.concat([best, scores], axis=-1)
        best = paddle.topk(combined, min(k, combined.shape[-1]), axis=-1)[0]
    return paddle.mean(best, axis=-1)


def acceptance_lengths(posterior_mask, vis_scores=None, k=7, candidates=None, predictions=None, pst=False):
    """Accept contiguous exact/relaxed matches, matching loose_match.py."""
    allowed = posterior_mask.astype("bool")
    length = allowed.shape[-1]
    if length == 0:
        return paddle.zeros([allowed.shape[0]], dtype="int64")
    if vis_scores is not None:
        scores = vis_scores[:, :length]
        # argsort twice returns each position's rank without a Python/GPU sync per token.
        rank = paddle.argsort(paddle.argsort(scores, axis=-1, stable=True), axis=-1)
        allowed = allowed | (rank < max(1, min(k, length)))
    if pst:
        if candidates is None or predictions is None:
            raise ValueError("PST requires candidate tokens and target predictions")
        allowed = allowed | paddle.any(predictions.unsqueeze(-1) == candidates.unsqueeze(1), axis=-1)
    return paddle.sum(paddle.cumprod(allowed.astype("int64"), dim=-1), axis=-1)


@on_device("text_model.norm.weight", keyword="model")
def _draft_tree(model, cache, root, buffers):
    prefix = cache.length
    paths = buffers["paths"]
    device = model.text_model.embed_tokens.weight.place
    tokens = [int(root)]
    parent_logits = {}
    parent_top = {}
    for depth in range(max(buffers["positions"]) + 1):
        indices = [i for i, path in enumerate(paths) if len(path) == depth]
        for i in indices:
            path = paths[i]
            if path:
                parent = path[:-1]
                logits = parent_logits[parent]
                rank = path[-1]
                if rank >= logits.shape[-1]:
                    raise ValueError("Tree rank exceeds draft vocabulary")
                if parent not in parent_top:
                    width = max(p[-1] for p in paths if p and p[:-1] == parent) + 1
                    parent_top[parent] = paddle.topk(logits, width)[1].tolist()
                token = int(parent_top[parent][rank])
                tokens.append(token)
        end = indices[-1] + 1
        mask = np.concatenate([np.ones((len(indices), prefix), dtype=bool), buffers["mask"][indices, :end]], axis=1)
        output = model(
            input_ids=paddle.to_tensor([[tokens[i] for i in indices]], dtype="int64", place=device),
            cache=cache,
            position_ids=paddle.to_tensor([prefix + depth] * len(indices), dtype="int64", place=device),
            tree_mask=None if mask.all() else paddle.to_tensor(mask, place=device),
        )
        for j, i in enumerate(indices):
            parent_logits[paths[i]] = output.logits[0, j]
    return tokens


@on_device("text_model.norm.weight", keyword="model")
def _draft_chain(model, cache, root, count):
    """Keep greedy candidates on device; the final token's KV is filled lazily."""
    token = paddle.to_tensor([[root]], dtype="int64")
    tokens = [token]
    for _ in range(count - 1):
        output = model(input_ids=token, cache=cache, logits_to_keep=1)
        token = paddle.topk(output.logits[0, -1], 1)[1].reshape([1, 1])
        tokens.append(token)
    return paddle.concat(tokens, axis=1)


def _trim(tokens, remaining, eos):
    tokens = tokens[:remaining]
    for i, token in enumerate(tokens):
        if token in eos:
            return tokens[:i + 1], True
    return tokens, False


@paddle.no_grad()
@on_device("text_model.norm.weight", index=1, keyword="model")
def generate(inputs, model, draft_model=None, method="lvspec", max_new_tokens=512,
             drop_rate=0.9, loose_hidden_k=7, tree_choices=None, eos_token_id=None,
             use_chain_fastpath=True, use_cuda_graph=True):
    if method not in ("ar", "specvlm_uniform", "lvspec", "lvspec_with_pst"):
        raise ValueError(f"Unknown method: {method}")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    ids = inputs["input_ids"]
    if ids.shape[0] != 1 or ids.shape[-1] == 0:
        raise ValueError("Expected a nonempty prompt with batch_size=1")
    models = [model] if method == "ar" else [model, draft_model]
    if method != "ar" and draft_model is None:
        raise ValueError("Speculative decoding requires a draft model")
    if draft_model is not None and model.config.vocab_size != draft_model.config.vocab_size:
        raise ValueError("Target and draft must use the same token vocabulary")
    model.eval()
    if draft_model is not None:
        draft_model.eval()
    eos_value = model.raw_config.get("eos_token_id", getattr(model.config, "eos_token_id", None)) if eos_token_id is None else eos_token_id
    eos = set(eos_value if isinstance(eos_value, (list, tuple)) else [eos_value]) - {None}
    synchronize(models)
    start = time.perf_counter()
    if max_new_tokens == 0:
        with paddle.device.device_guard(device_of(ids)):
            return dict(output_ids=ids.clone(), generate_len=0, decoding_time=0.0, inference_time=0.0, mean_accept_length=0.0)
    if tree_choices is None:
        from tree_choices.choices import chain, mc_sim_7b_63
        tree_choices = mc_sim_7b_63 if method == "specvlm_uniform" else chain
    buffers = generate_tree_buffers(tree_choices) if method != "ar" else None
    extra = len(buffers["paths"]) if buffers else 1
    cache = model.new_cache(ids.shape[-1] + max_new_tokens + extra)
    output = model(**inputs, cache=cache, logits_to_keep=1)
    root = int(paddle.argmax(output.logits[0, -1]).item())
    reference_hidden = output.hidden_states if method.startswith("lvspec") else None
    if method != "ar":
        draft_inputs = drop_visual_tokens_uniform(inputs, drop_rate, model.video_token_id)
        draft_cache = draft_model.new_cache(draft_inputs["input_ids"].shape[-1] + max_new_tokens + extra)
        draft_model(**draft_inputs, cache=draft_cache, logits_to_keep=1)
    synchronize(models)
    decode_start = time.perf_counter()
    generated, accepted_lengths = [], []
    draft_graph = None
    reference_chunks = None
    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        if method == "ar":
            accepted, stop = _trim([root], remaining, eos)
            generated.extend(accepted)
            if stop or len(generated) == max_new_tokens:
                break
            token = paddle.to_tensor([[root]], dtype="int64", place=ids.place)
            output = model(input_ids=token, cache=cache, logits_to_keep=1)
            root = int(paddle.argmax(output.logits[0, -1]).item())
            continue
        target_prefix, draft_prefix = cache.length, draft_cache.length
        if use_chain_fastpath and buffers["greedy_chain"]:
            draft_attention = draft_model.text_model.layers[0].self_attn
            graph_supported = (use_cuda_graph and draft_attention.mode == "sdpa" and draft_attention.head_dim >= 64
                and draft_model.text_model.embed_tokens.weight.dtype in (paddle.float16, paddle.bfloat16)
                and device_of(draft_model.text_model.embed_tokens.weight).startswith("gpu:")
                and len(getattr(draft_model, "devices", (0,))) == 1)
            if graph_supported:
                if draft_graph is None:
                    from .draft_graph import DraftGraph
                    draft_graph = DraftGraph(draft_model, draft_cache, root, extra)
                candidates = draft_graph.replay(root, draft_prefix)
            else:
                candidates = _draft_chain(draft_model, draft_cache, root, extra)
            output = model(input_ids=move(candidates, model.text_model.embed_tokens.weight), cache=cache)
            predictions = paddle.argmax(output.logits[0], axis=-1)
            candidates_on_target = move(candidates, predictions)
            predicted = predictions[:-1].unsqueeze(0)
            scores = None
            if reference_hidden is not None:
                if reference_chunks is None:
                    # Prompt hidden states are fixed for the whole request.
                    # Prepare lazily here so this work stays inside decode timing.
                    reference_chunks = _normalized_visual_chunks(reference_hidden, output.hidden_states)
                scores = visual_similarity_hidden_state(reference_hidden, output.hidden_states[:, :-1],
                                                        normalized_reference=reference_chunks)
            lengths = acceptance_lengths(predicted == candidates_on_target[:, 1:], scores,
                loose_hidden_k, candidates_on_target, predicted, pst=method == "lvspec_with_pst")
            # Read the decision, candidates and next-root predictions in one
            # transfer instead of synchronizing for length, tokens and root.
            decision = paddle.concat([lengths.reshape([-1]), candidates_on_target.reshape([-1]),
                                      predictions.reshape([-1])]).tolist()
            length = decision[0] + 1
            accepted, stop = _trim(decision[1:1 + min(length, remaining)], remaining, eos)
            count = len(accepted)
            cache.truncate(target_prefix + count)
            if count == extra:
                # Only a completely accepted chain needs the final candidate KV.
                # Its logits were unnecessary for constructing this block.
                draft_model(input_ids=candidates[:, -1:], cache=draft_cache, logits_to_keep=1)
            draft_cache.truncate(draft_prefix + count)
            generated.extend(accepted)
            accepted_lengths.append(count)
            root = decision[1 + extra + count - 1]
            if stop:
                break
            continue
        tokens = _draft_tree(draft_model, draft_cache, root, buffers)
        output = model(
            input_ids=paddle.to_tensor([tokens], dtype="int64", place=ids.place), cache=cache,
            position_ids=paddle.to_tensor([target_prefix + p for p in buffers["positions"]], dtype="int64", place=ids.place),
            tree_mask=paddle.to_tensor(buffers["mask"], place=ids.place),
        )
        predictions = paddle.argmax(output.logits[0], axis=-1)
        best_path, best_length = None, -1
        for path in buffers["retrieve_indices"]:
            idx = paddle.to_tensor(path[:-1], dtype="int64", place=predictions.place)
            predicted = paddle.index_select(predictions, idx).unsqueeze(0)
            candidates = paddle.to_tensor([[tokens[i] for i in path]], dtype="int64", place=predictions.place)
            exact = predicted == candidates[:, 1:]
            scores = None
            if reference_hidden is not None:
                hidden_idx = paddle.to_tensor(path[:-1], dtype="int64", place=output.hidden_states.place)
                hidden = paddle.index_select(output.hidden_states[0], hidden_idx).unsqueeze(0)
                scores = move(visual_similarity_hidden_state(reference_hidden, hidden), predictions)
            length = int(acceptance_lengths(exact, scores, loose_hidden_k, candidates, predicted,
                                            pst=method == "lvspec_with_pst")[0].item())
            if length > best_length:
                best_path, best_length = path, length
        chosen = best_path[:best_length + 1]
        accepted, stop = _trim([tokens[i] for i in chosen], remaining, eos)
        chosen = chosen[:len(accepted)]
        cache.compact([target_prefix + i for i in chosen], target_prefix)
        draft_cache.compact([draft_prefix + i for i in chosen], draft_prefix)
        generated.extend(accepted)
        accepted_lengths.append(len(accepted))
        root = int(predictions[chosen[-1]].item())
        if stop:
            break
    synchronize(models)
    end = time.perf_counter()
    with paddle.device.device_guard(device_of(ids)):
        suffix = paddle.to_tensor([generated], dtype="int64", place=ids.place)
        output_ids = paddle.concat([ids, suffix], axis=-1)
    return dict(output_ids=output_ids, generate_len=len(generated),
                decoding_time=end - decode_start, inference_time=end - start,
                mean_accept_length=float(np.mean(accepted_lengths)) if accepted_lengths else 0.0)
