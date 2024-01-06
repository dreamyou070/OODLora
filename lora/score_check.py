import os, argparse

def main(args) :

    parent = os.path.split(args.record_dir)[0]
    record_dir = os.path.join(parent, 'score_record_csv')
    os.makedirs(record_dir, exist_ok=True)

    files = os.listdir(args.record_dir)

    for file in files:
        file_name, file_ext = os.path.splitext(file)
        epoch_info = file_name.split('_')[-1]
        elems = []
        file_path = os.path.join(args.record_dir, file)
        with open(file_path, 'r') as f:
            content = f.readlines()
        score_dict = {}
        for line_ in content:
            line = line_.strip()
            first_elem = ['epoch', 'class_name', 'img_name']
            #first_elem = ['class_name', 'img_name']
            elem = [epoch_info]
            line = line_.strip()
            line_list = line.split(' | ')
            class_name = line_list[0].strip()
            img_name = line_list[1].strip()
            elem.append(class_name)
            elem.append(img_name)
            score_list = line_list[2:]
            for score_elem in score_list:
                name = score_elem.split('=')[0]
                score = score_elem.split('=')[1]
                first_elem.append(name)
                elem.append(score)
                if name not in score_dict.keys():
                    score_dict[name] = float(score.strip())
                else :
                    score_dict[name] += float(score.strip())
            if first_elem not in elems:
                elems.append(first_elem)
            elems.append(elem)

        new_elem = [['', '', 'total']]
        for k in score_dict.keys():
            new_elem[0].append(str(score_dict[k]))
        # ---------------------------------------------- make csv file ---------------------------------------------- #
        with open(os.path.join(record_dir, f'score_epoch_{epoch_info}.csv'), 'w') as f:
            for elem in elems:
                f.write(','.join(elem))
                f.write('\n')
        with open(os.path.join(record_dir, f'score_epoch_{epoch_info}.csv'), 'a') as f:
            for elem in new_elem:
                f.write(','.join(elem))
                f.write('\n')


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_dir', type=str,
                        default=r'../result/MVTec3D-AD_experiment/bagel/lora_training/res_16_up_down_text_embedding/score_record')
    args = parser.parse_args()
    main(args)
