import json
import csv
import os

domain_tag = 'domain_name'
main_image_id_tag = 'main_image_id'
item_id_tag = 'item_id'
item_name_tag = 'item_name'
image_id_tag = 'image_id'
height_tag = 'height'
width_tag = 'width'
path_tag = 'path'


def filter(json_obj):
    if domain_tag in json_obj:
        return json_obj[domain_tag] == 'amazon.com'
    return False


def get_itemname_for_object(obj):
    if item_name_tag in obj:
        names = obj[item_name_tag]
        for name_obj in names:
            if name_obj['language_tag'] == 'en_US':
                return name_obj.get('value', None)
    return None

def get_filepath_for_object(obj, image_list):
    if not main_image_id_tag in obj:
        return None
    main_image_id = obj[main_image_id_tag]
    main_image = image_list[main_image_id]
    image_folder = "data/abo/images/small"
    image_filepath = os.path.join(image_folder, main_image[path_tag])
    return image_filepath
    

def load_objects():

    listings_filepath = "data/abo/listings/metadata/listings_0.json"
    
    obj_data = []

    with open(listings_filepath, 'r') as f:
        for line in f:
            json_obj = json.loads(line)
            if filter(json_obj):
                obj_data.append(json_obj)
    return obj_data
    
def load_images():
    imagedata_filepath = "data/abo/images/metadata/images.csv"
    
    image_list = {}
    with open (imagedata_filepath, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            if image_id_tag in row:
                image_id = row[image_id_tag]
                image_list[image_id] = row

    return image_list