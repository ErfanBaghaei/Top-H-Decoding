OUTPUT=./teseet.json

python3 -u alpaca_generate.py \
  --save_address "$OUTPUT" \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --max_new_tokens 1024 \
  --temperature 2.0 \
  --top_p 0.9 \
  --do_sample \
  --item_cap 100 \
  --alpha 0.4

alpaca_eval --annotators_config 'gpt4' --model_outputs "$OUTPUT"
