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
        if cat == args.trg_cat:
            cat_dir = os.path.join(base_folder, f'{cat}')

            train_dir = os.path.join(cat_dir, 'train')
            test_dir = os.path.join(cat_dir, 'test')
            validation_dir = os.path.join(cat_dir, 'validation')

            train_ex_dir = os.path.join(cat_dir, 'train_ex')
            test_ex_dir = os.path.join(cat_dir, 'test_ex')
            os.makedirs(train_ex_dir, exist_ok=True)
            os.makedirs(test_ex_dir, exist_ok=True)
            # -------------------------------------------------------------------------------------------------------
            # (1) train
            good_train_dir = os.path.join(train_dir, 'good/rgb')
            sam_train_dir = os.path.join(train_dir, f'good/gt')
            os.makedirs(sam_train_dir, exist_ok=True)
            images = os.listdir(good_train_dir)
            for image in images:
                img_dir = os.path.join(good_train_dir, image)
                np_img = np.array(Image.open(img_dir))
                predictor.set_image(np_img)

                h, w, c = np_img.shape

                input_box = np.array([int(h / 5), int(w / 5), int(h * 4 / 5), int(w * 4 / 5)])

                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False, )

                mask_dict = {}
                for i, mask in enumerate(masks):
                    np_mask = (mask * 1)
                    np_mask = np.where(np_mask == 1, 1, 0) * 255
                    sam_result_pil = Image.fromarray(np_mask.astype(np.uint8))
                    sam_result_pil.save(os.path.join(sam_train_dir, image))
            # -------------------------------------------------------------------------------------------------------
            # (2) test
            good_test_dir = os.path.join(test_dir, 'good/rgb')
            sam_test_dir = os.path.join(test_dir, f'good/gt')
            os.makedirs(sam_test_dir, exist_ok=True)
            images = os.listdir(good_test_dir)
            for image in images:
                img_dir = os.path.join(good_test_dir, image)
                np_img = np.array(Image.open(img_dir))

                predictor.set_image(np_img)
                h, w, c = np_img.shape

                input_box = np.array([int(h / 5), int(w / 5), int(h * 4 / 5), int(w * 4 / 5)])

                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False, )

                mask_dict = {}
                for i, mask in enumerate(masks):
                    np_mask = (mask * 1)
                    np_mask = np.where(np_mask == 1, 1, 0) * 255
                    sam_result_pil = Image.fromarray(np_mask.astype(np.uint8))
                    sam_result_pil.save(os.path.join(sam_test_dir, image))

            # -------------------------------------------------------------------------------------------------------
            # (3) validation
            good_validation_dir = os.path.join(validation_dir, 'good/rgb')
            sam_validation_dir = os.path.join(validation_dir, f'good/gt')
            os.makedirs(sam_validation_dir, exist_ok=True)
            images = os.listdir(good_validation_dir)
            for image in images:
                img_dir = os.path.join(good_validation_dir, image)
                np_img = np.array(Image.open(img_dir))
                predictor.set_image(np_img)
                h, w, c = np_img.shape

                input_box = np.array([int(h / 5), int(w / 5), int(h * 4 / 5), int(w * 4 / 5)])

                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False, )

                mask_dict = {}
                for i, mask in enumerate(masks):
                    np_mask = (mask * 1)
                    np_mask = np.where(np_mask == 1, 1, 0) * 255
                    sam_result_pil = Image.fromarray(np_mask.astype(np.uint8))
                    sam_result_pil.save(os.path.join(sam_validation_dir, image))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='foam')
    args = parser.parse_args()
    main(args)