# Top-H-Decoding
## Abstract
Large language models (LLMs), despite their impressive performance across a wide range of tasks, often struggle to balance two competing objectives in open-ended text generation: fostering diversity and creativity while preserving logical coherence. Existing truncated sampling techniques, including temperature scaling, top-p (nucleus) sampling, and min-p sampling, aim to manage this trade-off. However, they exhibit limitations, particularly in the effective incorporation of the confidence of the model into the corresponding sampling strategy. For example, min-p sampling relies on a single top token as a heuristic for confidence, eventually underutilizing the information of the probability distribution. Toward effective incorporation of the confidence of the model, in this paper, we present \textbf{\textit{top-H}} decoding. We first establish the theoretical foundation of the interplay between creativity and coherence in truncated sampling by formulating an \textbf{entropy-constrained minimum divergence} problem. We then prove this minimization problem to be equivalent to an \textbf{entropy-constrained mass maximization} (ECMM) problem, which is NP-hard. Finally, we present top-H decoding, a computationally efficient greedy algorithm to solve the ECMM problem. Extensive empirical evaluations demonstrate that top-H outperforms the state-of-the-art (SoTA) alternative of min-p sampling by up to $\mathbf{25.63}\%$ on creative writing benchmarks, while maintaining robustness on question-answering datasets such as GPQA, GSM8K, and MT-Bench. Additionally, an \textit{LLM-as-judge} evaluation confirms that top-H indeed produces coherent outputs even at higher temperatures, where creativity is especially critical. In summary, top-H advances SoTA in open-ended text generation and can be \textbf{ easily integrated} into creative writing applications.
## Get Started
1. Install requirements.
   ```pip install -r requirements.txt```
2. To get the results for Alpaca-Eval, run the alpaca_evaluate.sh file. ```bash alpaca_evaluate.sh```


## Citation
## Contact

If you have any questions or want to use the code, feel free to contact:

Current:
* Erfan Baghaei Potraghloo (baghaeip@usc.edu)
* Seyedarmin Azizi (seyedarm@usc.edu)
   
