import os

score_folder = r'C:\Users\hpuser\Desktop\수연\[연구2]\MVTec3D-AD_Experiment_SDXL\cable_gland\vae_student_model_score'
score_files = os.listdir(score_folder)
totals = [['epoch','bent','good','thred']]
for score_file in score_files:
    name, ext = os.path.splitext(score_file)
    epoch = name.split('_')[-1]
    score_file_dir = os.path.join(score_folder, score_file)
    with open(score_file_dir, 'r') as f:
        content = f.readlines()
    elem = [int(epoch)]
    for line in content :
        line = line.strip()
        line = line.split(' : ')
        if len(line) == 2:
            cat, score = line[0], line[1]
            elem.append(score)
    totals.append(elem)
import csv
with open('1_test.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(totals)