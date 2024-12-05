# Modified using claude 3.5 sonnet!!
# Reduced the conversion time by ~50x with 64 cores.

import sys
from pathlib import Path
from os import walk
from os import path
from PIL import Image
import json
import tqdm
import multiprocessing as mp

def print_usage():
    print('convert_images_to_json [params] images_path output_path')
    print('--caption_extension')
    print('--num_processes')

def write_entry(args):
    folder, image_path, caption_path, image_filename, intern_imgs_path = args
    # open the file containing the prompt
    with open(caption_path) as prompt_file:
        prompt = prompt_file.read()
    
    # read the images info
    image = Image.open(image_path)
    image_width = image.width
    image_height = image.height
    ratio = image_height / image_width
    
    entry = {
        'width': image_width,
        'height': image_height,
        'ratio': ratio,
        'path': image_filename,
        'prompt': prompt,
        'sharegpt4v': ''
    }
    
    # make sure to copy the image to the internimgs folder with the new filename!
    image_output_path = intern_imgs_path.joinpath(image_filename)
    image.save(image_output_path)
    
    return entry

def main():
    args = sys.argv
    if len(args) < 3:
        print_usage()
        return
    
    input_folder = args[1]
    output_folder = args[-1]
    caption_extension = '.txt'
    num_processes = mp.cpu_count()  # Default to number of CPU cores
    
    try:
        caption_arg = args.index('--caption_extension')
        caption_extension = args[caption_arg + 1]
    except ValueError:
        pass
    
    try:
        num_processes_arg = args.index('--num_processes')
        num_processes = int(args[num_processes_arg + 1])
    except ValueError:
        pass
    
    # create a folder with the output path
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # create InternData and InternImgs inside the output path
    intern_data_folder = output_folder.joinpath('InternData')
    intern_data_folder.mkdir(parents=True, exist_ok=True)
    intern_imgs_folder = output_folder.joinpath('InternImgs')
    intern_imgs_folder.mkdir(parents=True, exist_ok=True)
    
    # create a data_info.json inside InternData
    data_info_path = intern_data_folder.joinpath('data_info.json')
    
    tasks = []
    for (dirpath, dirnames, filenames) in walk(input_folder):
        for filename in tqdm.tqdm(filenames, desc=f'Creating Tasks'):
            if not caption_extension in filename:
                continue
            # check if an image exists for this caption
            image_filename = filename[:-len(caption_extension)]
            for image_extension in ['.jpg', '.png', '.jpeg', '.webp', '.JPEG', '.JPG']:
                image_path = Path(dirpath).joinpath(image_filename + image_extension)
                if path.exists(image_path):
                    tasks.append((dirpath, image_path, Path(dirpath).joinpath(filename), image_filename + image_extension, intern_imgs_folder))
                    break
    
    # process tasks using multiprocessing
    with mp.Pool(num_processes) as pool:
        json_entries = list(tqdm.tqdm(pool.imap(write_entry, tasks), total=len(tasks)))
    
    # write the entries to the json file
    with open(data_info_path, 'w') as json_file:
        json.dump(json_entries, json_file)

if __name__ == '__main__':
    main()