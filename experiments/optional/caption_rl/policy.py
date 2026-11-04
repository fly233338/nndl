"""使用同一策略完成 SCST 采样和采样序列的概率重评分。"""

import torch


@torch.no_grad()
def rollout(decoder, memory, bos_id, eos_id, max_length, sample):
    """按当前策略逐词采样或贪心生成，结束后用 EOS 补齐批次。"""
    tokens = torch.full((memory.size(0), 1), bos_id, device=memory.device, dtype=torch.long)
    finished = torch.zeros(memory.size(0), device=memory.device, dtype=torch.bool)
    for _ in range(max_length - 1):
        logits = decoder._decode(memory, tokens)[:, -1]
        next_token = (
            torch.multinomial(logits.softmax(-1), 1) if sample else logits.argmax(-1, keepdim=True)
        )
        next_token = torch.where(finished[:, None], torch.full_like(next_token, eos_id), next_token)
        tokens = torch.cat((tokens, next_token), dim=1)
        finished |= next_token[:, 0].eq(eos_id)
        if finished.all():
            break
    return tokens


def policy_loss(decoder, features, sampled, advantage, eos_id):
    """重算采样序列的对数概率，并用停止梯度的奖励优势加权。"""
    # eval() 仅关闭 dropout，保证采样和重评分使用一致的策略。
    logits = decoder(features, sampled[:, :-1])
    log_probs = logits.log_softmax(-1).gather(-1, sampled[:, 1:, None]).squeeze(-1)
    # 首次 EOS 是有效动作，结束后强制补齐的 EOS 不计入损失。
    active = sampled[:, :-1].eq(eos_id).cumsum(dim=1).eq(0)
    return -(advantage.detach() * (log_probs * active).sum(dim=1)).mean()
