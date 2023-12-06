import os
import argparse
def extract_pure_name(img_dir) :
    parent, img_name = os.path.split(img_dir)
    name, ext = os.path.splitext(img_name)
    return name, ext

def main(args) :

    print(f' step 1. call generated image folder')
    generated_image_folder = args.generated_image_folder
    sample_dir = os.path.join(generated_image_folder, 'sample')
    images = os.listdir(sample_dir)
    for image in images :
        image_dir = os.path.join(sample_dir, image)
        name, ext = os.path.splitext(image_dir)
        epoch = name.split('_')[-2]
        epoch = int(epoch.split('e')[-1])

        epoch_dir = os.path.join(sample_dir, str(epoch))
        os.makedirs(epoch_dir, exist_ok=True)
        new_dir = os.path.join(epoch_dir, image)
        os.rename(image_dir, new_dir)




if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_image_folder',
                        default='./result/haibara_experience/one_image/name_3_without_caption/haibara_base')
    args = parser.parse_args()
    main(args)
