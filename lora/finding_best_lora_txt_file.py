import os
import argparse
def main(args) :

    files = os.lsitdir(args.folder)
    for file in files :
        lora_epoch = file.split('_')[2]
        score_diffs = 0
        with open(os.path.join(args.folder, file), 'r') as f :
            content = f.readlines()
            for line in content :
                line = line.strip()
                if len(line) > 0 :
                    class_name, img_name, anormal_score, normal_score = line.sptlie(' | ')
                    anormal_score = float(anormal_score.replace('anormal_score = ',''))
                    normal_score = float(anormal_score.replace('normal_score = ',''))
                    score_diff = normal_score - anormal_score
                    score_diffs += score_diff
        print(f'lora_epoch : {lora_epoch}, score_diffs : {score_diffs}')




if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='../result/MVTec3D-AD_experiment/bagel/lora_training/2_training_16res_with_CrossEntropy/anormality_score/test')
    args = parser.parse_args()
    main(args)
