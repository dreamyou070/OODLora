import argparse, os
from PIL import Image
def main(args) :

    print(f'step 1. image dir')
    img_folder = args.img_folder
    images = os.listdir(img_folder)
    for image in images :
        img_dir = os.path.join(img_folder, image)
        pil_img = Image.open(img_dir)
        print(pil_img.size)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', type=str, default=r'C:\data7\sooyeon\medical_image\experiment_data\MV\bagel\train\good\rgb')
    args = parser.parse_args()
    main(args)