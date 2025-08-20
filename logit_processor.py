import torch
import torch.nn.functional as F
from transformers import LogitsProcessor


class TopH_LogitsProcessor(LogitsProcessor):

    """
    [`LogitsProcessor`] that implements Top-H sampling, a decoding method which adaptively selects a subset of
    high-probability tokens based on entropy and cumulative probability constraints.

    This method dynamically determines how many tokens to keep by analyzing the entropy difference of the selected
    distribution, thereby balancing exploration and exploitation. It ensures that generated text maintains both
    diversity and coherence.

    Args:
        top_n (`int`, *optional*, defaults to 100):
            The maximum number of tokens to consider for filtering. 
            Only the top `top_n` tokens (by probability) are evaluated.
        temperature (`float`, *optional*, defaults to 1.0):
            Softmax temperature. Higher values increase randomness, while lower values make predictions sharper.
        alpha (`float`, *optional*, defaults to 0.4):
            Scaling coefficient for the entropy-based threshold (`tau`). Must be in the range `(0, 1]`.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> from my_module import TopH_LogitsProcessor

    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    >>> inputs = tokenizer("The quick brown fox", return_tensors="pt")

    >>> logits_processor = TopH_LogitsProcessor(top_n=50, temperature=0.9, alpha=0.5)
    >>> outputs = model.generate(**inputs, logits_processor=[logits_processor])
    >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    The quick brown fox jumps over the lazy dog.
    ```
    """


    def __init__(self, top_n = 100,temperature=1.0, alpha = 0.4):
        super().__init__()

        # --- input checks ---
        if temperature == 0:
            raise ValueError("Temperature must be non-zero to perform Top-H decoding.")
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in the range (0, 1].")
        
        self.top_n = top_n
        self.temperature = temperature
        self.coef = alpha

    @staticmethod
    def calculate_entropy(probs):

        """
        Computes Shannon entropy of a probability distribution.

        Args:
            probs (`torch.FloatTensor`):
                Probability distribution over tokens.

        Return:
            `torch.FloatTensor`: Scalar entropy value.
        """

        probs = probs/torch.sum(probs) 
        return -torch.sum(probs * torch.log2(probs))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                
        """
        Filters logits using Top-H sampling.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input token IDs.
            scores (`torch.FloatTensor` of shape `(batch_size, vocab_size)`):
                Raw logits from the model.

        Return:
            `torch.FloatTensor` of shape `(batch_size, vocab_size)`:
                Processed logits where invalid tokens are masked with `-inf`.
        """
                
        batch_size, vocab_size = scores.shape
        assert batch_size == 1, "This implementation handles a single example at a time."

        # --- compute probabilities ---
        scaled_logits = scores
        probs = F.softmax(scaled_logits/self.temperature, dim=-1)  

        # --- extract top-N candidates ---
        torch.set_printoptions(precision=8)
        top_n_probs, top_n_indices = torch.topk(probs[0], self.top_n, largest=True, sorted=True)  

        # --- entropy-based thresholding ---
        alpha = top_n_probs.sum()
        tau = ((self.calculate_entropy(top_n_probs) - torch.log2(alpha)) * alpha - bias ) * self.coef 

        valid_indices = []
        ind = 1
        sigma = top_n_probs[0]
        H = - top_n_probs[0] * torch.log2(top_n_probs[0])

        for idx, prob in zip(top_n_indices, top_n_probs):
            valid_indices.append(idx)
            ind += 1

            # Update cumulative sums
            H -= top_n_probs[ind-1] * torch.log2(top_n_probs[ind-1])
            sigma += top_n_probs[ind-1]
            entropy_diff = ((H/sigma)+torch.log2(sigma)) 

            # Stopping criterion: stop when entropy exceeds threshold
            if entropy_diff > (tau/sigma + torch.log2(sigma)) :
                break

        # --- build mask of valid tokens ---                
        keep_mask = torch.zeros(vocab_size, dtype=torch.bool)
        keep_mask[torch.tensor(valid_indices, dtype=torch.long, device=scores.device)] = True

        # --- apply filtering ---
        updated_scores = scaled_logits.clone()
        updated_scores[:, ~keep_mask] = float('-inf')
        
        return updated_scores
