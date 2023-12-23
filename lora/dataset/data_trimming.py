import torch, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import argparse
import shutil

def main(args):

    print(f'\n step 1. make model')
    #device = args.device


    print(f'\n step 2. dataset')
    base_folder = 'C:/Users/hpuser/Desktop/수연/[연구2]/MVTec3D-AD_Experiment_SDXL'
    data_folder = args.data_folder
    classes = os.listdir(data_folder)

    for cls in classes:
        if cls != 'cable_gland' :
            test_mask_folder = os.path.join(data_folder, cls, 'data/test_mask')
            org_folder = os.path.join(data_folder, cls, 'data/test/bad')
            base_folder_dir = os.path.join(base_folder, cls)
            inpainted_folder = os.path.join(base_folder_dir, 'corrected')
            new_folder = os.path.join(base_folder_dir, 'corrected_new')
            os.makedirs(new_folder, exist_ok=True)

            sub_classes = os.listdir(org_folder)
            for sub_cls in sub_classes:
                repeat_num, sub = sub_cls.split('_')
                if sub != 'good' :

                    sub_org_folder = os.path.join(org_folder, sub_cls)
                    sub_inpainted_folder = os.path.join(inpainted_folder, sub)
                    new_sub_folder = os.path.join(new_folder, sub_cls)
                    os.makedirs(new_sub_folder, exist_ok=True)
                    images = os.listdir(sub_org_folder)
                    for image in images:
                        name, ext = image.split('.')
                        org_image_path = os.path.join(sub_org_folder, image)
                        org_pil = Image.open(org_image_path)
                        org_pil = org_pil.resize((512, 512))
                        org_pil.save(os.path.join(new_sub_folder, image))
                        np_org = np.array(org_pil)

                        inpainted_image_path = os.path.join(sub_inpainted_folder, image)
                        inpainted_pil = Image.open(inpainted_image_path).resize((512, 512))
                        inpainted_pil.save(os.path.join(new_sub_folder, f'{name}_inpainted.{ext}'))
                        np_inpainted = np.array(inpainted_pil)

                        mask_image_path = os.path.join(test_mask_folder, sub, image)
                        mask_pil = Image.open(mask_image_path).convert('RGB')
                        mask_pil = mask_pil.resize((512, 512))
                        mask_pil.save(os.path.join(new_sub_folder, f'{name}_mask.{ext}'))


                        #np_mask = np.array(mask_pil)
                        #np_mask = np.where(np_mask < 100, 0, 1) # black = 0 = background, white = 1

                        #print(f'np_org : {np_org.shape}')
                        #print(f'np_inpainted : {np_inpainted.shape}')
                        #print(f'np_mask : {np_mask.shape}')

                        #new_np = np_org * (1-np_mask) + np_inpainted * (np_mask)
                        #new_pil = Image.fromarray(new_np.astype(np.uint8))
                        #new_pil.save(os.path.join(new_sub_folder, f'{name}_new.{ext}'))
                else :
                    sub_org_folder = os.path.join(org_folder, sub_cls)
                    new_sub_folder = os.path.join(new_folder, sub_cls)
                    shutil.copytree(sub_org_folder, new_sub_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str, default='C:/Users/hpuser/Desktop/수연/[연구2]/MVTec3D-AD_Experiment')
    args = parser.parse_args()
    main(args)