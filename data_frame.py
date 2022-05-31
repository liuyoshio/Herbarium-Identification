import pandas as pd
import json

def getDataFrame():
    train_dir = '../herbarium-2022-fgvc9/train_images/'

    with open('../herbarium-2022-fgvc9/train_metadata.json') as json_file:
        train_meta = json.load(json_file)

    image_ids = [image["image_id"] for image in train_meta["images"]]
    image_dirs = [train_dir + image['file_name'] for image in train_meta["images"]]
    category_ids = [annotation['category_id'] for annotation in train_meta['annotations']]
    genus_ids = [annotation['genus_id']-1 for annotation in train_meta['annotations']]

    train_df = pd.DataFrame({
        "image_id" : image_ids,
        "image_dir" : image_dirs,
        "category" : category_ids,
        "genus" : genus_ids})
    
    
    return train_df