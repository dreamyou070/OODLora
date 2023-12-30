from segment_anything import SamPredictor, sam_model_registry
import argparse, os
from PIL import Image
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
        cat_dir = os.path.join(base_folder, f'{cat}/train_ex/bad')
        folders = os.listdir(cat_dir)
        for folder in folders:
            folder_dir = os.path.join(cat_dir, folder)
            images = os.listdir(folder_dir)
            for image in images:
                img_dir = os.path.join(folder_dir, image)
                pil_img = Image.open(img_dir)
                np_img = np.array(pil_img)
                predictor.set_image(np_img)
                masks, _, _ = predictor.predict(cat)
                print(f'masks : {masks}')
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    args = parser.parse_args()
    main(args)