from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from translation.manager import Manager


def triu_mask(size: int, device: str | None = None) -> Tensor:
    mask = torch.ones((1, size, size), device=device)
    return torch.triu(mask, diagonal=1) == 0


def greedy_search(
    manager: 'Manager', src_encs: Tensor, max_length: int = 512
) -> tuple[Tensor, Tensor]:
    model, vocab, device = manager.model, manager.vocab, manager.device
    tgt_mask = triu_mask(max_length, device=device)
    path = torch.full((1, max_length), vocab.BOS, device=device)
    prob = torch.zeros((1, max_length), device=device)

    for i in range(1, max_length):
        tgt_encs = model.decode(src_encs, path[:, :i], tgt_mask=tgt_mask[:, :i, :i])
        logits = model.out_embed(tgt_encs[:, -1], inverse=True)
        scores = logits.log_softmax(dim=-1).max(dim=-1)
        prob[0, i], path[0, i] = scores
        if path[0, i] == vocab.EOS:
            return path[0, : i + 1], prob[0, : i + 1]

    return path[0], prob


def beam_search(
    manager: 'Manager', src_encs: Tensor, beam_size: int = 4, max_length: int = 512
) -> tuple[Tensor, Tensor]:
    model, vocab, device = manager.model, manager.vocab, manager.device
    tgt_mask = triu_mask(max_length, device=device)
    active = torch.ones(beam_size, dtype=torch.bool, device=device)
    paths = torch.full((beam_size, max_length), vocab.BOS, device=device)
    probs = torch.zeros(beam_size, device=device)

    probs_ = torch.zeros((beam_size, max_length), device=device)

    i, init_size = 0, beam_size
    while (i := i + 1) < max_length and beam_size > 0:
        tgt_encs = model.decode(
            src_encs.expand(beam_size, -1, -1), paths[active, :i], tgt_mask=tgt_mask[:, :i, :i]
        )
        logits = model.out_embed(tgt_encs[:, -1], inverse=True)
        scores = probs[active].unsqueeze(1) + logits.log_softmax(dim=-1)
        if i == 1:
            scores = scores[0]

        topv, topi = torch.topk(scores.flatten(), beam_size)
        if beam_size < init_size:
            active_clone = active.clone()
            active_clone[~active] |= probs[~active] < topv.max() / i
            active = active_clone
            active_count = int(active.count_nonzero())
            if active_count > beam_size:
                beam_size = active_count
                topv, topi = torch.topk(scores.flatten(), beam_size)

        reorder = topi // vocab.size()
        paths[active] = paths[active][reorder]
        paths[active, i] = topi % vocab.size()
        probs[active] = topv

        probs_[active] = probs_[active][reorder]
        probs_[active, i] = logits.log_softmax(dim=-1).flatten()[topi]

        terminated = paths[:, i] == vocab.EOS
        probs[terminated] = probs[terminated] / i
        active = active & ~terminated
        beam_size = int(active.count_nonzero())

    argmax = probs.argmax()
    return paths[argmax, :i], probs_[argmax, :i]
