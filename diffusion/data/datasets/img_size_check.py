import multiprocessing
from tqdm import tqdm
import imagesize
import json

def process_image_pair(dct):
    w1, h1 = imagesize.get(dct['source_image'])
    w2, h2 = imagesize.get(dct['edited_image'])
    return 1 if w1 != w2 or h1 != h2 else 0

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def main():
    # Assuming 'Data' is your list of dictionaries
    Data = read_jsonl("")
    
    # Number of processes to use
    num_processes = multiprocessing.cpu_count()
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use tqdm to show progress
        results = list(tqdm(pool.imap(process_image_pair, Data), total=len(Data)))
    
    # Sum up the results to get the count of mismatched pairs
    c = sum(results)
    
    print(f"Number of mismatched image pairs: {c}")

if __name__ == '__main__':
    main()