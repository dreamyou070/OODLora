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
        print(f'cat : {cat} : trg_cat : {args.trg_cat}')
        if cat == args.trg_cat:
            print(f'cat : {cat}')

            cat_dir = os.path.join(base_folder, f'{cat}')

            train_ex_dir = os.path.join(cat_dir, 'train_ex')

            train_rgb_dir = os.path.join(train_ex_dir, 'mask')
            train_gt_dir = os.path.join(train_ex_dir, 'gt')
            train_pixel_mask_dir = os.path.join(train_ex_dir, 'mask')
            os.makedirs(train_pixel_mask_dir, exist_ok=True)

            folders = os.listdir(train_rgb_dir)
            for folder in folders:
                print(f'folder : {folder}')
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
                        trg_h_1, trg_w_1 = h / 2, w * (5 / 11)
                        trg_h_2, trg_w_2 = h * (1 / 2), w * (6 / 11)
                        input_point = np.array([[trg_h_1, trg_w_1], [trg_h_2, trg_w_2]])
                        input_label = np.array([1, 1])
                        masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,
                                                                  multimask_output=True, )

                        for i, (mask, score) in enumerate(zip(masks, scores)):
                            if i == 1:
                                np_mask = (mask * 1)
                                np_mask = np.where(np_mask == 1, 1, 0) * 255  # if true,  be black
                                sam_result_pil = Image.fromarray(np_mask.astype(np.uint8))
                                sam_result_pil.save(os.path.join(mask_folder_dir, image))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='foam')
    args = parser.parse_args()
    main(args)