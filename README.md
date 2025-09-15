# inference
'''
python unified_image_editing_inference.py \
    --model_path /workspace/models/b-ours-v1 \
    --generation_mode edit \
    --image_path /workspace/bagel/dataset/demo/demo_sample/image/260/first_frame.jpg \
    --edit_prompt "You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> <|vision_start|> image will be generated here <|vision_end|>

When you want to generate an image, use the special tokens <|vision_start|> and <|vision_end|> to indicate where the image should be placed.'
User: Draw what it will look like one hour later."''
" \
    --use_autoregressive \
    --max_length 300
‘’‘


