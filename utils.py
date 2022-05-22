import torch 

SPECIAL_TOKENS = ["[bos]", "[eos]", "[dyn]", "[img]", "[txt]"] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]", 'additional_special_tokens': ["[dyn]", "[img]", "[txt]"],}


def accuracy_compute(logits, labels, top_k=5, ignore_index=-100): 
    bsz, seq_len, _ = logits.size()
    logits = logits.contiguous().view(bsz*seq_len, -1)
    _, idx = torch.topk(logits, top_k, -1) 
    correct = idx.eq(labels.view(-1, 1).expand_as(idx)) 
    correct_total = correct.view(-1).float().sum().item()
    nums = labels.view(-1).detach().cpu().numpy()
    length = 0 
    for num in nums:
        if num != ignore_index:
            length += 1
    return correct_total / float(length) 
