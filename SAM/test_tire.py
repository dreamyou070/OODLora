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
                trg_h_0, trg_w_00 = h / 4, w * (5 / 12)
                trg_h_0, trg_w_01 = h / 4, w * (6 / 12)
                trg_h_0, trg_w_02 = h / 4, w * (7 / 12)

                trg_h_0 = h * (1 / 4)
                trg_h_01 = h * (2 / 5)
                trg_h_02 = h * (4 / 5)
                trg_h_1 = h / 2
                trg_h_2 = h * (3/ 4)

                trg_w_01 = w * (4 / 12)
                trg_w_02 = w * (5 / 12)


                input_point = np.array([[trg_h_0, trg_w_01],[trg_h_0, trg_w_02],
                                        [trg_h_01, trg_w_01],[trg_h_01, trg_w_02],
                    [trg_h_0, trg_w_00],[trg_h_0, trg_w_01], [trg_h_0, trg_w_02],
                                        [trg_h_1, trg_w_00],[trg_h_1, trg_w_01], [trg_h_1, trg_w_02],
                                        [trg_h_2, trg_w_00],[trg_h_2, trg_w_01], [trg_h_2, trg_w_02],
                                        [trg_h_02, trg_w_01], [trg_h_02, trg_w_02],
                                        [trg_h_2, trg_w_01],[trg_h_2, trg_w_02]])
                input_label = np.array([1, 1, 1,
                                        1, 1, 1,
                                        1, 1, 1,1,1,1,1,1,1,1,1])
                masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,
                                                          multimask_output=True, )
                mask_dict = {}
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if i == 1 :
                        np_mask = (mask * 1)
                        np_mask = np.where(np_mask == 1, 0, 1) #* 255  # if true,  be black
                        mask_dict[1] = np_mask
                    if i == 2 :
                        np_mask = (mask * 1)
                        np_mask = np.where(np_mask == 1, 0, 1) #* 255
                        mask_dict[2] = np_mask
                final_mask = (1- (mask_dict[1] * mask_dict[2])) * 255
                sam_result_pil = Image.fromarray(final_mask.astype(np.uint8))
                sam_result_pil.save(f'mask_{image}')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='foam')
    args = parser.parse_args()
    main(args)