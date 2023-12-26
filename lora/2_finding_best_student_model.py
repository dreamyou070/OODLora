import os, argparse, yaml, torch, cv2, numpy as np
from PIL import Image

def main(args) :

    inference_folder = args.inference_folder
    epochs = os.listdir(inference_folder)
    total_diff = []
    print(f'epochs : {epochs}')
    for epoch in epochs :
        epoch_num = epoch.split('_')[-1]
        epoch_elem = [epoch_num]
        epoch_dir = os.path.join(inference_folder, epoch)
        test_dir = os.path.join(epoch_dir, 'test_dataset')
        cats = os.listdir(test_dir)
        title_list = ['epoch']
        title_list += cats
        if title_list not in total_diff :
            total_diff.append(title_list)
        for cat in cats :
            test_cat_dir = os.path.join(test_dir, cat)
            images = os.listdir(test_cat_dir)
            cat_diff = []
            for image in images :
                name, ext = os.path.splitext(image)

                if 'recon' not in name and 'gt' not in name and 'mask' not in name :
                    base_img_dir = os.path.join(test_cat_dir, image)
                    recon_img_dir = os.path.join(test_cat_dir, f'{name}_recon{ext}')
                    base_pil = Image.open(base_img_dir).convert('RGB').resize((512, 512))
                    base_np = np.array(base_pil)

                    recon_pil = Image.open(recon_img_dir).convert('RGB').resize((512, 512))
                    recon_np = np.array(recon_pil)

                    if 'good' not in cat :
                        mask_dir = os.path.join(test_cat_dir, f'{name}_mask{ext}')
                        mask_pil = Image.open(mask_dir).convert('RGB').resize((512, 512))
                        mask_np = np.array(mask_pil)
                        mask_np = np.where(mask_np > 100, 1, 0)
                    else :
                        mask_np = np.zeros((512,512,3))

                    diff = (base_np - recon_np) **2

                    background_diff = diff * mask_np
                    background_pixel_num = 512*512 - np.sum(mask_np)
                    background_diff = np.sum(background_diff/ background_pixel_num)

                    object_diff = diff * (1-mask_np)
                    object_pixel_num = np.sum(mask_np)
                    if object_pixel_num > 0 :
                        object_diff = np.sum(object_diff/ object_pixel_num)
                        cat_diff.append(float(background_diff - object_diff))
                    else :
                        cat_diff.append(float(background_diff))
                    cat_diff_score = np.mean(np.array(cat_diff))
                    print(f'cat_diff : {cat_diff_score}')
                    epoch_elem.append(cat_diff_score)
        total_diff.append(epoch_elem)

    import csv
    parent, child = os.path.split(inference_folder)
    csv_file_dir = os.path.join(parent, f'{child}_scoring.csv')
    with open(csv_file_dir, 'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(total_diff)
    print(f'csv file saved at {csv_file_dir}')



if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_folder', type=str,
                        default ='../result/MVTec3D-AD_experiment/cable_gland/vae_training/1_TS_encoder_normal_abnormal_aug/inference_finding_best_epoch')

    args = parser.parse_args()
    main(args)