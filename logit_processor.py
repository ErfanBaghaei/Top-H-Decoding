import torch
import torch.nn.functional as F
from transformers import LogitsProcessor


class TopH_LogitsProcessor(LogitsProcessor):
    def __init__(self, top_n = 100,temperature=1.0, alpha = 0.4):
        super().__init__()
        self.top_n = top_n
        self.temperature = temperature
        self.coef = alpha

    @staticmethod
    def calculate_entropy(probs):
        probs = probs/torch.sum(probs) 
        return -torch.sum(probs * torch.log2(probs))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape
        assert batch_size == 1, "This implementation handles a single example at a time."
        bias = 0.0
        scaled_logits = scores
        probs = F.softmax(scaled_logits/self.temperature, dim=-1)  
        torch.set_printoptions(precision=8)
        top_n_probs, top_n_indices = torch.topk(probs[0], self.top_n, largest=True, sorted=True)  
        alpha = top_n_probs.sum()
        tau = ((self.calculate_entropy(top_n_probs) - torch.log2(alpha)) * alpha - bias ) * self.coef 
        valid_indices = []
        ind = 1
        sigma = top_n_probs[0]
        H = - top_n_probs[0] * torch.log2(top_n_probs[0])
        for idx, prob in zip(top_n_indices, top_n_probs):
            valid_indices.append(idx)
            ind += 1
            H -= top_n_probs[ind-1] * torch.log2(top_n_probs[ind-1])
            sigma += top_n_probs[ind-1]
            entropy_diff = ((H/sigma)+torch.log2(sigma)) 
            if entropy_diff > (tau/sigma + torch.log2(sigma)) :
                break
                
        keep_mask = torch.zeros(vocab_size, dtype=torch.bool)
        keep_mask[torch.tensor(valid_indices, dtype=torch.long, device=scores.device)] = True
        updated_scores = scaled_logits.clone()
        updated_scores[:, ~keep_mask] = float('-inf')
        return updated_scores
