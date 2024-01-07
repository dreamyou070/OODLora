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

            train_ex_dir = os.path.join(cat_dir, 'train_ex')
            train_ex_rgb_dir = os.path.join(train_ex_dir, 'rgb')
            train_ex_gt_dir = os.path.join(train_ex_dir, 'gt')

            test_ex_dir = os.path.join(cat_dir, 'test_ex')
            test_ex_rgb_dir = os.path.join(test_ex_dir, 'rgb')
            test_ex_gt_dir = os.path.join(test_ex_dir, 'gt')
            # -------------------------------------------------------------------------------------------------------
            # (1) train
            class_names = os.listdir(train_ex_rgb_dir)
            for class_name in class_names:
                if 'good' in class_name:
                    class_rgb_dir = os.path.join(train_ex_rgb_dir, class_name)
                    class_gt_dir = os.path.join(train_ex_gt_dir, class_name)
                    os.makedirs(class_gt_dir, exist_ok=True)
                    images = os.listdir(class_rgb_dir)
                    for image in images:
                        img_dir = os.path.join(class_rgb_dir, image)
                        np_img = np.array(Image.open(img_dir))
                        predictor.set_image(np_img)
                        input_point = np.array([[200, 200]])
                        input_label = np.array([0])
                        masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,
                                                                  multimask_output=True, )
                        for i, (mask, score) in enumerate(zip(masks, scores)):
                            np_mask = (mask * 1)
                            np_mask = np.where(np_mask == 1, 1, 0) * 255
                            sam_result_pil = Image.fromarray(np_mask.astype(np.uint8))
                            sam_result_pil.save(f'{i}_{image}')


                        break
            """
            # -------------------------------------------------------------------------------------------------------
            # (2) test
            class_names = os.listdir(test_ex_rgb_dir)
            for class_name in class_names:
                if 'good' in class_name:
                    class_rgb_dir = os.path.join(test_ex_rgb_dir, class_name)
                    class_gt_dir = os.path.join(test_ex_gt_dir, class_name)
                    os.makedirs(class_gt_dir, exist_ok=True)
                    images = os.listdir(class_rgb_dir)
                    for image in images:
                        img_dir = os.path.join(class_rgb_dir, image)
                        np_img = np.array(Image.open(img_dir))
                        predictor.set_image(np_img)
                        input_point = np.array([[0, 0]])
                        input_label = np.array([0])
                        masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,
                                                                  multimask_output=True, )
                        for i, (mask, score) in enumerate(zip(masks, scores)):
                            if len(masks) < 3:
                                print('wrong')
                            else:
                                if i == 2:
                                    np_mask = (mask * 1)
                                    np_mask = np.where(np_mask == 1, 0, 1) * 255
                                    sam_result_pil = Image.fromarray(np_mask.astype(np.uint8))
                                    sam_result_pil.save(os.path.join(class_gt_dir, image))
            """
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--trg_cat', type=str, default='foam')
    args = parser.parse_args()
    main(args)