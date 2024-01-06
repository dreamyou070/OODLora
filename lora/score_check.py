import os, argparse

def main(args) :

    files = os.listdir(args.record_dir)
    print(f'files: {files}')

    for file in files:
        file_name, file_ext = os.path.splitext(file)
        epoch_info = file_name.split('_')[-1]
        elems = []
        file_path = os.path.join(args.record_dir, file)
        with open(file_path, 'r') as f:
            content = f.readlines()
        print(f'content[-1] : {content[-1]}')
        for line_ in content:
            line = line_.strip()
            print(f'line: {line}')
            #first_elem = ['epoch', 'class_name', 'img_name']
            first_elem = ['class_name', 'img_name']
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
            if first_elem not in elems:
                elems.append(first_elem)
            elems.append(elem)
        # ---------------------------------------------- make csv file ---------------------------------------------- #
        csv_file = os.path.join(args.record_dir, f'score_epoch_{epoch_info}.csv')
        with open(csv_file, 'w') as f:
            for elem in elems:
                f.write(','.join(elem))
                f.write('\n')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_dir', type=str,
                        default=r'../result/MVTec3D-AD_experiment/bagel/lora_training/res_64_up_down_text_embedding/score_record')
    args = parser.parse_args()
    main(args)
