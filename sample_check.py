import os
save_folder = r'10_good'
os.makedirs(save_folder, exist_ok=True)
base_folder = r'example'
folders = os.listdir(base_folder)
global_num = 0
for folder in folders:
    folder_path = os.path.join(base_folder, folder)
    images = os.listdir(folder_path)
    for image in images:
        image_path = os.path.join(folder_path, image)

        a = str(global_num).zfill(3)
        new_path =  os.path.join(save_folder, f'{a}.jpg')
        global_num += 1
        os.rename(image_path, new_path)
