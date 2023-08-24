import json
import csv
import os

domain_tag = 'domain_name'
main_image_id_tag = 'main_image_id'
item_id_tag = 'item_id'
item_name_tag = 'item_name'
item_keywords_tag = 'item_keywords'
image_id_tag = 'image_id'
height_tag = 'height'
width_tag = 'width'
path_tag = 'path'

def is_keyword(sub_string, full_string):
    return all(char in full_string for char in sub_string)
    
def filter(json_obj):
    name = get_itemname_for_object(json_obj)
    if name is None:
        return False
    #print(name)
    
    if not main_image_id_tag in json_obj:
        return False
    
    keywords = get_keywords_for_object(json_obj)
    #if keywords is None:
    #    return False
    #print(keywords)
    key = 'drink'
    kw = 'drink'
    dotcom = json_obj[domain_tag] == 'amazon.com'
    if domain_tag in json_obj:
        return dotcom#(key in name or kw in keywords)
    return False


def get_keywords_for_object(obj):
    if item_keywords_tag in obj:
        keywords = obj[item_keywords_tag]
        return [kw['value'] for kw in keywords]
    return None
    
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
    

def load_objects(num_objects = None):
    suffixes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    
    obj_data = []

    for s in suffixes:
        listings_filepath = f"data/abo/listings/metadata/listings_{s}.json"

        with open(listings_filepath, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                if filter(json_obj):
                    obj_data.append(json_obj)
                if num_objects is not None and len(obj_data) >= num_objects:
                    return obj_data
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