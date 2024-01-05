import os

record_dir = r'../result/MVTec3D-AD_experiment/bagel/lora_training/res_64_32_up_down_text_embedding/per_res_normalized_cross_attention_map/score_record/score_epoch_1.txt'
with open(record_dir, 'r') as f :
    content = f.readlines()[0]
content = content.split('\n')
for line in content :
    line_list = line.split(' | ')
    class_name = line_list[0]
    img_name = line_list[1]
    print(f'class_name: {class_name}, img_name: {img_name}')
    score_list = line_list[2:]
    for score_elem in score_list :
        print(score)
        name = score_elem.split('=')[0]
        score = score_elem.split('=')[1]
    
    break

# combined | 001.png | res_64_down_attn_0=839 | res_64_down_attn_1=1528 | res_32_down_attn_0=3121 | res_32_down_attn_1=3650 | res_32_up_attn_0=2963 | res_32_up_attn_1=3659 | res_32_up_attn_2=3959 | res_64_up_attn_0=14341 | res_64_up_attn_1=12120 | res_64_up_attn_2=8650 | down_64=1329 | down_32=3819 | up_32=3810 | up_64=11452