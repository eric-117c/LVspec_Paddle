import torch


def loose_match_window(posterior_mask, loose_window=2):
    if isinstance(posterior_mask, torch.Tensor) is False:
        posterior_mask = torch.tensor(posterior_mask)

    accept_lengths = []
    for row in posterior_mask:
        count = 0
        idx = 0
        while idx < len(row):
            if row[idx] == 1:
                count += 1
                idx += 1
            elif row[idx] == 0:
                #loose window
                loose_count = 0
                while idx + loose_count + 1 < len(row) and row[idx + loose_count + 1] == 1:
                    loose_count += 1

                if loose_count >= loose_window:
                    count += loose_count + 1  
                    idx += loose_count + 1
                else:
                    break
            else:
                idx += 1

        accept_lengths.append(count)
    return torch.tensor(accept_lengths)




def loose_match_random(posterior_mask, accept_prob: float = 0.5):
    if not isinstance(posterior_mask, torch.Tensor):
        posterior_mask = torch.tensor(posterior_mask)

    accept_lengths = []
    for row in posterior_mask:   # row: [T]
        count = 0
        idx = 0
        while idx < len(row):
            if row[idx] == 1:
                count += 1
                idx += 1

            elif row[idx] == 0:
                if torch.rand(1).item() < accept_prob:
                    count += 1
                    idx += 1
                else:
                    break

            else:
                idx += 1

        accept_lengths.append(count)

    return torch.tensor(accept_lengths, device=posterior_mask.device)



def loose_match_entropy(posterior_mask, entropies, entropy_threshold: float = 0.6):
    if not isinstance(posterior_mask, torch.Tensor):
        posterior_mask = torch.tensor(posterior_mask)

    if not isinstance(entropies, torch.Tensor):
        entropies = torch.tensor(entropies, dtype=torch.float32, device=posterior_mask.device)
    else:
        entropies = entropies.to(posterior_mask.device, dtype=torch.float32)

    if entropies.dim() == 1:
        entropies = entropies.unsqueeze(0).expand(posterior_mask.size(0), -1)

    accept_lengths = []
    B, T = posterior_mask.shape

    for b in range(B):
        row = posterior_mask[b]     # [T]
        ent_row = entropies[b]      # [T]
        count = 0
        idx = 0
        while idx < T:
            if row[idx] == 1:
                count += 1
                idx += 1

            elif row[idx] == 0:
                if ent_row[idx] > entropy_threshold:
                    count += 1
                    idx += 1
                else:
                    break

            else:
                idx += 1

        accept_lengths.append(count)

    return torch.tensor(accept_lengths, device=posterior_mask.device)





def loose_match_window_and_entropy(
    posterior_mask,
    entropies,
    entropy_threshold: float = 0.1,
    loose_window: int = 2,
):
    if not isinstance(posterior_mask, torch.Tensor):
        posterior_mask = torch.tensor(posterior_mask)

    if not isinstance(entropies, torch.Tensor):
        entropies = torch.tensor(entropies, dtype=torch.float32, device=posterior_mask.device)
    else:
        entropies = entropies.to(posterior_mask.device, dtype=torch.float32)

    B, T = posterior_mask.shape
    # assert entropies.shape == (B, T), f"entropies shape {entropies.shape} must match posterior_mask {posterior_mask.shape}"

    accept_lengths = []

    for b in range(B):
        row = posterior_mask[b]   # [T]
        ent_row = entropies[b]    # [T]
        count = 0
        idx = 0

        while idx < T:
            if row[idx] == 1:
                count += 1
                idx += 1

            elif row[idx] == 0:
                if ent_row[idx] <= entropy_threshold:
                    break

                loose_count = 0
                while idx + loose_count + 1 < T and row[idx + loose_count + 1] == 1:
                    loose_count += 1

                if loose_count >= loose_window:
                    count += loose_count + 1
                    idx += loose_count + 1
                    print("accept by window + entropy")
                else:
                    break

            else:
                idx += 1

        accept_lengths.append(count)

    return torch.tensor(accept_lengths, device=posterior_mask.device)





def loose_match_hidden_topk_or_search(
    posterior_mask: torch.Tensor,   # [B, T_match]
    vis_scores: torch.Tensor,       # [B, T_match]
    candidates: torch.Tensor,       # [B, T_cand]
    logits: torch.Tensor,           # [B, T_match, V]
    k: int = 5,
):
    if not isinstance(posterior_mask, torch.Tensor):
        posterior_mask = torch.tensor(posterior_mask)
    posterior_mask = posterior_mask.to(logits.device)

    if not isinstance(vis_scores, torch.Tensor):
        vis_scores = torch.tensor(vis_scores, dtype=torch.float32, device=logits.device)
    else:
        vis_scores = vis_scores.to(logits.device, dtype=torch.float32)

    if not isinstance(candidates, torch.Tensor):
        candidates = torch.tensor(candidates, dtype=torch.long, device=logits.device)
    else:
        candidates = candidates.to(logits.device)

    B, T_match = posterior_mask.shape
    assert logits.shape[0] == B and logits.shape[1] >= T_match, \
        f"logits not match: {logits.shape} vs posterior_mask {posterior_mask.shape}"
    assert candidates.shape[0] == B, \
        f"candidates batch not match: {candidates.shape[0]} vs {B}"

    if vis_scores.dim() == 1:
        vis_scores = vis_scores.unsqueeze(0).expand(B, -1)
    assert vis_scores.shape[1] >= T_match, \
        f"vis_scores dim >= T_match: {vis_scores.shape[1]} vs {T_match}"
    vis_scores = vis_scores[:, :T_match]  

    k = max(1, min(k, T_match))               
    low_mask = torch.zeros_like(vis_scores, dtype=torch.bool)  # [B, T_match]

    for b in range(B):
        row_scores = vis_scores[b]               # [T_match]
        topk_indices = torch.argsort(row_scores, dim=-1)[:k]   # [k]
        low_mask[b, topk_indices] = True

    accept_lengths = []

    for b in range(B):
        row = posterior_mask[b]           # [T_match]
        allow_relax_hidden = low_mask[b]  # [T_match]
        cand_row = candidates[b]          # [T_cand]
        T_cand = cand_row.size(0)

        count = 0
        idx = 0

        while idx < T_match:
            if row[idx] == 1:
                count += 1
                idx += 1

            elif row[idx] == 0:
                cond_hidden = bool(allow_relax_hidden[idx].item())

                arg_token = torch.argmax(logits[b, idx], dim=-1).item()
                cond_search = (cand_row == arg_token).any().item()

                if cond_hidden or cond_search:
                    count += 1
                    idx += 1
                    src = []
                    if cond_hidden: src.append("top-k")
                    if cond_search: src.append("search")
                    print(f"[b={b}, idx={idx-1}] accept by {', '.join(src)}")
                else:
                    break

            else:
                idx += 1

        accept_lengths.append(count)

    return torch.tensor(accept_lengths, device=logits.device)

