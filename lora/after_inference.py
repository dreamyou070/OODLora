import os
import argparse
def main(args) :

    base_folder = args.base_folder
    cats = os.listdir(base_folder)

    for cat in cats:
        cat_dir = os.path.join(base_folder, f'{cat}')
        sub_cats = os.listdir(cat_dir)
        for sub_cat in sub_cats:
            sub_cat_dir = os.path.join(cat_dir, f'{sub_cat}')
            images = os.listdir(sub_cat_dir)
            for image in images:
                if args.trg_word in image :
                    org_image_path = os.path.join(sub_cat_dir, image)
                    os.remove(org_image_path)
if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--trg_word', type=str)
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD')
    args = parser.parse_args()
    main(args)
