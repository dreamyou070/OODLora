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
    trg_img_dir = os.path.join(base_folder, args.sub_dir)
    np_img = np.array(Image.open(trg_img_dir))
    #h, w, c = np_img.shape
    h, w = np_img.shape
    h_min, h_max = 0, int(h * 14/30)
    w_min, w_max = 0, int(w * 1/5)
    for h_ in range(h_min, h_max, 1):
        for w_ in range(w_min, w_max, 1):
            np_img[h_, w_] = 0
    new_mask = Image.fromarray(np_img)
    new_mask.save('test.png')
    """
    
    predictor.set_image(np_img)
    
    trg_h_1, trg_w_1 = h / 2, w * (4 / 11)
    trg_h_2, trg_w_2 = h * (1 / 2), w * (7 / 11)
    input_point = np.array([[trg_h_1, trg_w_1], [trg_h_2, trg_w_2]])
    input_label = np.array([1, 1])
    masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,
                                              multimask_output=True, )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        if i == 1:
            np_mask = (mask * 1)
            np_mask = np.where(np_mask == 1, 1, 0) * 255  # if true,  be black
            sam_result_pil = Image.fromarray(np_mask.astype(np.uint8))
            sam_result_pil.save('test.png')
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--sub_dir', type=str, default='rope/train_ex/mask/80_cut/cut_018.png')
    args = parser.parse_args()
    main(args)