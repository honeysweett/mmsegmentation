import json

def merge_coco_jsons(json_file1, json_file2, output_file):
    # Load JSON files
    with open(json_file1, 'r') as f:
        data1 = json.load(f)
    
    with open(json_file2, 'r') as f:
        data2 = json.load(f)

    # Merge images, annotations, and categories
    merged_images = data1['images'] + data2['images']
    merged_annotations = data1['annotations'] + data2['annotations']
    merged_categories = data1['categories'] + data2['categories']

    # Create new IDs for images and annotations to ensure they are continuous
    new_image_id_map = {}
    new_annotation_id_map = {}
    new_category_id_map = {}

    current_image_id = 1
    for image in merged_images:
        new_image_id_map[image['id']] = current_image_id
        image['id'] = current_image_id
        current_image_id += 1

    current_annotation_id = 1
    for annotation in merged_annotations:
        new_annotation_id_map[annotation['id']] = current_annotation_id
        annotation['id'] = current_annotation_id
        annotation['image_id'] = new_image_id_map[annotation['image_id']]
        current_annotation_id += 1

    current_category_id = 1
    for category in merged_categories:
        if category['id'] not in new_category_id_map:
            new_category_id_map[category['id']] = current_category_id
            category['id'] = current_category_id
            current_category_id += 1
        else:
            category['id'] = new_category_id_map[category['id']]

    merged_data = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories
    }

    # Save the merged JSON file
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

# File paths
# json_file1 = '/mmdetection/BSTLD_dataset/dataset_test_rgb/bstld_test_coco_format.json'
# json_file2 = '/mmdetection/MTSD_dataset/OpenDataLab___Mapillary_Traffic_Sign_Dataset/raw/mtsd_test_coco_format_2.json'
# output_file = '/mmdetection/tools_json/merged_test_dataset_3.json'

json_file1 = '/mmdetection/BSTLD_dataset/dataset_train_rgb/bstld_train_coco_format.json'
json_file2 = '/mmdetection/MTSD_dataset/OpenDataLab___Mapillary_Traffic_Sign_Dataset/raw/mtsd_train_coco_format_2.json'
output_file = '/mmdetection/tools_json/merged_train_dataset.json'

merge_coco_jsons(json_file1, json_file2, output_file)
