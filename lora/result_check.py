import os

base_folder = r'../result/MVTec3D-AD_experiment'
window_folder = r'../result/Window_MVTec3D-AD_experiment'
os.makedirs(window_folder, exist_ok=True)
classess = os.listdir(base_folder)
for class_ in classess:
    new_class_dir = os.path.join(window_folder, class_)
    os.makedirs(new_class_dir, exist_ok=True)
    vae_result_dir = os.path.join(new_class_dir, 'vae_result')
    os.makedirs(vae_result_dir, exist_ok=True)
    unet_result_dir = os.path.join(new_class_dir, 'unet_result')
    os.makedirs(unet_result_dir, exist_ok=True)

