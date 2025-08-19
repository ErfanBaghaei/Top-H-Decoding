python3 -u -m lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
  --tasks gpqa_main_generative_n_shot \
  --device cuda:0 \
  --batch_size 1 \
  --num_fewshot 8 \
  --gen_kwargs "top_p=1.0,temperature=2.0,do_sample=True"
