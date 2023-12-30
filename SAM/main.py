from segment_anything import SamPredictor, sam_model_registry
import argparse, os
from PIL import Image
import numpy as np

def main(args):

    print(f'step 1. prepare model')

    model_type = "vit_h"
    path_to_checkpoint= r'/home/dreamyou070/pretrained_stable_diffusion/sam_vit_h_4b8939.pth'
    sam = sam_model_registry[model_type](checkpoint=path_to_checkpoint)
    predictor = SamPredictor(sam)

    print(f'step 2. prepare images')
    base_folder = args.base_folder
    cats = os.listdir(base_folder)
    for cat in cats:
        if cat == 'bagel' :
            cat_dir = os.path.join(base_folder, f'{cat}/train_ex/bad')
            save_cat_dir = os.path.join(base_folder, f'{cat}/train_ex/bad_sam')
            os.makedirs(save_cat_dir, exist_ok=True)
            folders = os.listdir(cat_dir)
            for folder in folders:
                if 'good' in folder :
                    save_folder_dir = os.path.join(save_cat_dir, folder)
                    os.makedirs(save_folder_dir, exist_ok=True)

                    folder_dir = os.path.join(cat_dir, folder)
                    images = os.listdir(folder_dir)
                    for image in images:
                        img_dir = os.path.join(folder_dir, image)
                        np_img = np.array(Image.open(img_dir))
                        predictor.set_image(np_img)
                        input_point = np.array([[0,0]])
                        input_label = np.array([0])
                        masks, scores, logits = predictor.predict(point_coords=input_point,point_labels=input_label,multimask_output=True,)
                        for i, (mask, score) in enumerate(zip(masks, scores)):
                            if i == 1 :
                                np_mask = (mask * 1)
                                np_mask = np.where(np_mask == 1, 0, 1) * 255
                                Image.fromarray(np_mask.astype(np.uint8)).save(os.path.join(save_folder_dir, image))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    args = parser.parse_args()
    main(args)