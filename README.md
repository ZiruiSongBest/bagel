
```
python unified_image_editing_inference.py \
    --model_path /workspace/models/b-ours-v1 \
    --generation_mode edit \
    --image_path /workspace/bagel/dataset/demo/demo_sample/image/260/first_frame.jpg \
    --edit_prompt "You are a multimodal AI that can generate both text and images. When generating content:

1. For text: Simply generate text naturally
2. For images: Use the special tokens <|vision_start|> and <|vision_end|> to mark where an image should be generated

- Mixed response: "I'll create an image for you: <|vision_start|> <|vision_end|> And here's some additional text."

Always use <|vision_start|> and <|vision_end|> tokens when you want to generate an image.'
User: Draw what it will look like one hour later."''
" \
    --use_autoregressive \
    --max_length 300
```




## merge finetuned weights

First, consolidate the FSDP checkpoint into a single file:

```bash
torchrun --standalone --nproc_per_node=3 scripts/consolidate_fsdp_checkpoint.py \
  --checkpoint /workspace/bagel/results/unified_finetune_20250924_211049/checkpoints/0000010 \
  --model-path /workspace/models/BAGEL-7B-MoT \
  --output /workspace/bagel/results/unified_finetune_20250924_211049/checkpoints/0000010/ema.safetensors \
  --state ema \
```

Second, merge the consolidated checkpoint with the base model:

```bash
python scripts/merge_finetuned_weights.py \
  --target-dir /workspace/models/ours-ema-v4 \
  --base-ema /workspace/models/BAGEL-7B-MoT/ema.safetensors \
  --checkpoint /workspace/bagel/results/unified_finetune_20250924_211049/checkpoints/0000010/ema.safetensors \
  --output-name ema.safetensors
```