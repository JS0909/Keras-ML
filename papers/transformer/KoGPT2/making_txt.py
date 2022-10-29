import json
import glob
from tqdm import tqdm

# 폴더 내 json 파일 이름을 리스트로 저장
# for문을 돌려가며 각 json 파일 내 텍스트 부분을 텍스트파일로 저장. 엔터로 구분
# WBRW1900003252.json 까지 문학

path = 'D:/study_data/_data/team_project/korean_written/NIKL_WRITTEN(v1.1)/'
all_json_names = sorted(glob.glob(path + '*.json'))

print('number of files: ',len(all_json_names))
all_text = []
for this_file in tqdm(all_json_names):
    with open(this_file, "r", encoding='UTF8') as f:
        this_json = json.load(f)        
    dic_list = this_json['document'][0]['paragraph']
    this_text = [dic['form'] for dic in dic_list]
    all_text.append('\n'.join(this_text))

f = open('written.txt', 'w', encoding='UTF8')
for texts in tqdm(all_text):
    f.write(texts)

print('created txt.')