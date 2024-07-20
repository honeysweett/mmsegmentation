import json

def update_category_id(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for annotation in data['annotations']:
        if annotation['category_id'] == 1:
            annotation['category_id'] = 2

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# 文件路径
input_file = '/mmdetection/MTSD_dataset/OpenDataLab___Mapillary_Traffic_Sign_Dataset/raw/mtsd_train_coco_format.json'
output_file = '/mmdetection/MTSD_dataset/OpenDataLab___Mapillary_Traffic_Sign_Dataset/raw/mtsd_train_coco_format_2.json'

update_category_id(input_file, output_file)
