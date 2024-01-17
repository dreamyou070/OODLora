from segment_anything import SamPredictor, sam_model_registry
import argparse, os
from PIL import Image
import numpy as np


def main(args):

    print(f'step 1. prepare model')

    model_type = "vit_h"
    path_to_checkpoint = r'/home/dreamyou070/pretrained_stable_diffusion/sam_vit_h_4b8939.pth'
    sam = sam_model_registry[model_type](checkpoint=path_to_checkpoint)
    predictor = SamPredictor(sam)

    print(f'step 2. prepare images')
    base_folder = args.base_folder
    cats = os.listdir(base_folder)
    for cat in cats:
        if cat == args.trg_cat:
            cat_dir = os.path.join(base_folder, f'{cat}')
            train_ex_dir = os.path.join(cat_dir, 'train_ex')

            train_rgb_dir = os.path.join(train_ex_dir, 'rgb')
            train_gt_dir = os.path.join(train_ex_dir, 'gt')
            train_pixel_mask_dir = os.path.join(train_ex_dir, 'mask')
            os.makedirs(train_pixel_mask_dir, exist_ok=True)

            folders = os.listdir(train_rgb_dir)
            for folder in folders:
                if 'good' not in folder :
                    rgb_folder_dir = os.path.join(train_rgb_dir, folder)
                    mask_folder_dir = os.path.join(train_pixel_mask_dir, folder)
                    os.makedirs(mask_folder_dir, exist_ok=True)

                    images = os.listdir(rgb_folder_dir)
                    for image in images:
                        rgb_img_dir = os.path.join(rgb_folder_dir, image)
                        np_img = np.array(Image.open(rgb_img_dir))
                        predictor.set_image(np_img)
                        h, w, c = np_img.shape
                        input_point = np.array([[0,0]])
                        input_label = np.array([0])
                        masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,
                                                                  multimask_output=True, )
                        for i, (mask, score) in enumerate(zip(masks, scores)):
                            if i == 1:
                                np_mask = (mask * 1)
                                np_mask = np.where(np_mask == 1, 0, 1) * 255
                                sam_result_pil = Image.fromarray(np_mask.astype(np.uint8))
                                sam_result_pil = sam_result_pil.resize((512,512))
                                np_sample = np.array(sam_result_pil)
                                h, w, c = np_img.shape
                                min_h, min_2_h, max_h, max_2_h = 0, h/10, h/10*9, h
                                min_w, min_2_w, max_w, max_2_w = 0, w/10, w/10*9, w
                                for h_index in range(h):
                                    for w_index in range(w):
                                        if h_index < min_2_h :
                                            if w_index < min_2_w or w_index > max_2_w :
                                                np_sample[h_index, w_index,:] = 0
                                        elif h_index > max_2_h :
                                            if w_index < min_2_w or w_index > max_2_w :
                                                np_sample[h_index, w_index,:] = 0
                                Image.fromarray(np_sample.astype(np.uint8)).save(os.path.join(mask_folder_dir, image))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='dowel')
    args = parser.parse_args()
    main(args)