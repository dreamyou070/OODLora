from imgutils.metrics import ccip_difference, ccip_clustering
import os


# ccip_difference 가 작을 수록 유사한 이미지
def extract_pure_name(img_dir):
    parent, img_name = os.path.split(img_dir)
    name, ext = os.path.splitext(img_name)
    return name, ext


haibara_score_text = 'haibara_ccip_score.txt'
base_folder = '/data7/sooyeon/MyData/haibara_dataset/not_test/haibara_19/2_girl'
files = os.listdir(base_folder)
img_dirs = []
for file in files:
    file_dir = os.path.join(base_folder, file)
    name, ext = extract_pure_name(file_dir)
    if ext != '.txt':
        img_dirs.append(file_dir)

arisu_base_folder = '/data7/sooyeon/MyData/arisu/arisu_20/20_girl'
arisu_files = os.listdir(arisu_base_folder)
arisu_img_dirs = []
for file in arisu_files:
    file_dir = os.path.join(arisu_base_folder, file)
    name, ext = extract_pure_name(file_dir)
    if ext != '.txt':
        arisu_img_dirs.append(file_dir)

records = []
for img_dir_1 in img_dirs:
    for img_dir_2 in arisu_img_dirs:
        if img_dir_1 != img_dir_2:
            img_name_1, ext_1 = extract_pure_name(img_dir_1)
            img_name_2, ext_2 = extract_pure_name(img_dir_2)
            difference = ccip_difference(img_dir_1, img_dir_2)
            record = [img_name_1, img_name_2, difference]
            reverse_record = [img_name_2, img_name_1, difference]
            if record not in records and record not in records:
                records.append(record)
import csv

csv_file = 'haibara_arisu_ccip_score.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['img_name_1', 'img_name_2', 'ccip_difference'])
    for record in records:
        writer.writerow(record)


