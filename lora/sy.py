import os

record_dir = r'../result/MVTec3D-AD_experiment/bagel/lora_training/res_64_32_up_down_text_embedding/per_res_normalized_cross_attention_map/score_record/score_epoch_1.txt'
with open(record_dir, 'r') as f :
    content = f.readlines()
print(content)