import os

record_dir = r'../result/MVTec3D-AD_experiment/bagel/lora_training/res_64_32_up_down_text_embedding/per_res_normalized_cross_attention_map/score_record'
files = os.listdir(record_dir)


for file in files :
    file_name, file_ext = os.path.splitext(file)
    epoch_info = file_name.split('_')[-1]
    elems = []
    file_path = os.path.join(record_dir, file)
    with open(file_path, 'r') as f :
        content = f.readlines()
    for line_ in content :
        first_elem = ['epoch', 'class_name', 'img_name']
        elem = [epoch_info]
        line = line_.strip()
        line_list = line.split(' | ')
        class_name = line_list[0].strip()
        img_name = line_list[1].strip()
        elem.append(class_name)
        elem.append(img_name)
        score_list = line_list[2:]
        for score_elem in score_list :
            name = score_elem.split('=')[0]
            score = score_elem.split('=')[1]
            first_elem.append(name)
            elem.append(score)
        if first_elem not in elems :
            elems.append(first_elem)
        elems.append(elem)
    # ---------------------------------------------- make csv file ---------------------------------------------- #
    csv_file = f'../result/MVTec3D-AD_experiment/bagel/lora_training/res_64_32_up_down_text_embedding/score_epoch_{epoch_info}.csv'
    with open(csv_file, 'w') as f :
        for elem in elems :
            f.write(','.join(elem))
            f.write('\n')