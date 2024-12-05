import sys
from pathlib import Path
from os import walk
from os import path
import os
from PIL import Image
import json
from tqdm.auto import tqdm
import multiprocessing as mp
import imagesize

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data



def read_json(path):
    with open(path,'r') as f:
        x = json.load(f)
    return x


def process_data(dct):
    '''
    dct has : source_image, edited_image, insturction
    '''
    w1, h1 = imagesize.get(dct['source_image'])
    w2, h2 = imagesize.get(dct['edited_image'])
    if w1==w2 and h1==h2:
        ratio = h1/w1 
    else:
        ratio = 10.0 # pairs with ratio > 4.5 are filtered in data-set

    entry = {
        'width': w1,
        'height': h1,
        'ratio': ratio,
        'src_path': dct['source_image'].replace('ANON',''),
        'edit_path': dct['edited_image'].replace('ANON',''),
        'prompt': dct['instruction'],
        'sharegpt4v': ''
    }

    return entry


def process_cimnli(dct):
    root_dir = 'ANON'
    w1, h1 = imagesize.get(os.path.join(root_dir, 'images', dct['image_input_filename']))
    w2, h2 = imagesize.get(os.path.join(root_dir, 'images_c1', dct['image_output_filename']))

    ratio = h1/w1

    entry = {
        'width': w1,
        'height': h1,
        'ratio': ratio,
        'src_path': os.path.join('images',dct['image_input_filename']),
        'edit_path': os.path.join('images_c1', dct['image_output_filename']),
        'prompt': dct['question'],
        'sharegpt4v': ''
    }

    return entry

def main():
    save_data_path = 'ANON/pixart_format_3.3M.json'
    Data = read_jsonl("ANON/complete_train_3.3M.jsonl")
    num_processes = mp.cpu_count()

    with mp.Pool(processes=num_processes) as pool:
        # Use tqdm to show progress
        json_entries = list(tqdm(pool.imap(process_data, Data), total=len(Data)))


    with open(save_data_path,'w') as json_file:
        json.dump(json_entries, json_file, indent=4)


if __name__ == '__main__':
    main()