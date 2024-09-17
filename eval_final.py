import json
import os
import pandas as pd
from tabulate import tabulate
import re
from fuzzywuzzy import fuzz
import argparse


PROMPT_FORMAT = {
    
    "noreason_name": (
        f"### Instruction:\n{{instruction}}\n\n### Input:\n{{input}}\n\n### Moive Name:"
    ),
    
    "noreason_id": (
        f"### Instruction:\n{{instruction}}\n\n### Input:\n{{input}}\n\n### Moive ID:"
    ),
    
    "reason_name": (
        f"### Instruction:\n{{instruction}}\n\n### Input:\n{{input}}\n\n### Response:"
    ),
    
    "reason_id": (
        f"### Instruction:\n{{instruction}}\n\n### Input:\n{{input}}\n\n### Response:"
    ),
    
    }

MARKERS = {
    "noreason_name": "### Moive Name:",
    "noreason_id": "### Moive ID:",
    "reason_name": "### Response:",
    "reason_id": "### Response:",
    
}

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        jsonl = json.load(f)
    return jsonl

def load_predictions(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            predictions = json.load(f)
        return predictions
    else:
        return None

def remove_underscores(string):
    # Replace all underscores with nothing
    return string.replace("_", "")

def extract_after_marker(text, marker):
    # Define the marker
    # marker = "### Moive Name:"
    
    # Find the position of the marker
    index = text.find(marker)
    
    if index != -1:
        # Get the substring after the marker
        # Adding the length of the marker to the index moves the start to just after the marker
        movie_name = text[index + len(marker):].strip()
        return movie_name
    else:
        return None

def process_dataset(jsonl_dir, experiment:str):

    data_json = json.load(open(jsonl_dir))
    
    experiment = remove_after_last_underscore(experiment)
    
    # change format of prompt
    def apply_prompt_template(sample):
        return {
            "prompt": PROMPT_FORMAT[experiment].format(instruction=sample['instruction'], input=sample['input']),
            "generation": sample['output'],
        }
    formated_data = [apply_prompt_template(sample) for sample in data_json]

    return formated_data

def is_match_high(string1, string2, threshold=90):
    # Calculate the fuzzy match ratio
    match_ratio = fuzz.ratio(string1, string2)
    # Compare the match ratio with the threshold
    return match_ratio > threshold


def extract_digit_between_identifiers(long_string):
    '''
    return : a string of digits (didn't covnert to int)
    
    '''
    # Define the pattern
    pattern = r"###(\d{1,2})###"


    # Find all matches
    matches = re.findall(pattern, long_string)
    return matches

def extract_content_between_identifiers(long_string):
    # Define the special identifier
    identifier = "###"
    
    # Find the positions of the start and end of the special identifier
    start_idx = long_string.find(identifier)
    end_idx = long_string.find(identifier, start_idx + len(identifier))
    
    # If both identifiers are found, extract the content between them
    if start_idx != -1 and end_idx != -1:
        # Include the length of the identifier to skip to the end of the first identifier
        start_idx += len(identifier)
        # Extract the content between the identifiers
        content = long_string[start_idx:end_idx]
        return content
    else:
        # Return None if identifiers are not found properly
        return None

def remove_after_last_underscore(s):
    # Find the index of the last underscore
    index = s.rfind('_')
    # If underscore is found, slice the string up to that index
    if index != -1:
        return s[:index]
    # If no underscore is found, return the original string
    return s


def is_movie_in_database(movie_name, search_table):
    # Check if the movie name (in lowercase) is in the search table
    return movie_name.lower() in search_table



def clean_movie_titles(title):
    # Regular expression to find a four-digit year within parentheses
    cleaned_title = re.sub(r'\s*\(\d{4}\)', '', title).strip()
    return cleaned_title


def calculate_match_ratio(gt_outputs, outputs, generation_type, id_or_name, experiment_type, test_or_eval, search_table, num_candidates, args, origin_data):
    '''
    match ratio: a list contains 0 and 1s, 1 means the generated movie name is in the ground truth list
        - like [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, ...]
    
    '''
    experiment_type = remove_after_last_underscore(experiment_type)
    match_ratio = []
    error = 0
    halluciation = 0
    for gt_output, output, origin_datapoint in zip(gt_outputs, outputs, origin_data):
        predict = extract_after_marker(output, MARKERS[experiment_type]) # prediction text
        if predict:
            # process
            if generation_type == 'reason':
                if test_or_eval == 'eval':
                    if id_or_name == 'name':
                        gt_output = extract_content_between_identifiers(gt_output)
                    else:
                        gt_output = extract_digit_between_identifiers(gt_output)
        
                if id_or_name == 'name':
                    predict = extract_content_between_identifiers(predict)
                else:
                    predict = extract_digit_between_identifiers(predict)
            ### check if predict is valid
            if not predict:
                error += 1
                continue  
            else:
                if id_or_name == "id":
                    predict = predict[0] # extract the first element if the lst is not empty
                else:
                    pass # keep the string if name
            
            # compute match
            if args.consider_retrieval_fail and origin_datapoint['retrieval_success'] == 0:
                match_ratio.append(0)
                continue
            
            # compute match
            if id_or_name == 'name':
                predict = clean_movie_titles(predict)
                success = 0
                for gt in gt_output:
                    if fuzz.ratio(gt, predict)>90:
                        success = 1
                        break
                
                match_ratio.append(success)
            else:  # id
                success = 0
                for gt in gt_output:
                    try:
                        if int(gt) == int(predict):
                            success = 1
                            break
                    except:
                        pass
                match_ratio.append(success)
                
            # compute halluciation
            if id_or_name == 'name':
                predict = clean_movie_titles(predict)
                if not is_movie_in_database(predict, search_table):
                    halluciation += 1
            else: # generate id
                try:
                    generate_id = int(predict)
                except:
                    error += 1
                    continue
                if generate_id < 1 or generate_id > num_candidates:
                    halluciation += 1
        else:
            error += 1
    print(f" Error ratio for {generation_type}, {experiment_type}, {test_or_eval} :", error/len(outputs))
    print(f"Halluciation for {generation_type}, {experiment_type}, {test_or_eval} :", halluciation/(len(outputs)-error))
    print(f"match ratio for {generation_type}, {experiment_type}, {test_or_eval} :", sum(match_ratio) / len(match_ratio))
    return sum(match_ratio) / len(match_ratio), halluciation/(len(outputs)-error)



def create_movie_search_table(file_path, column_name='primaryTitle'):
    movie_names_set = set()
    chunk_size = 5000
    for chunk in pd.read_csv(file_path, sep='\t', usecols=[column_name], chunksize=chunk_size):
        movie_names_set.update(chunk[column_name].str.lower().dropna().unique())
    return movie_names_set

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the results of the hallucination task")
    parser.add_argument("--consider_retrieval_fail", action="store_true", help="Consider retrieval failure")
    parser.add_argument("--retriever", type=str, default="openai", choices = ["bge15", "openai"], help="Retriever used for retrieval")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_names = ['inspired', 'redial']
    generation_types = ['reason']
    test_eval_settings = ['test']
    id_name_settings = ['name'] # ['id', 'name']
    num_candidates_list = [10, 50]

    imdb_tsv_dir = "/data_hdd/dayu/RALLM_CRS/hallucination/names.tsv"
    movie_table = create_movie_search_table(imdb_tsv_dir)

    results = []

    for dataset_name in dataset_names:
        for generation_type in generation_types:
            for test_eval_setting in test_eval_settings:
                for id_name in id_name_settings:
                    for num_candidates in num_candidates_list:
                        #eval_path = f"/usa/dayu/CRS/RA-CRS/Re-rank/own_reranker/test_results/eval_{generation_type}_{id_name}_{test_eval_setting}_{num_candidates}.json"
                        test_path = f"/usa/dayu/CRS/RA-CRS/Re-rank/own_reranker/test_results/test_{generation_type}_{id_name}_{test_eval_setting}_{num_candidates}_{dataset_name}_{args.retriever}.json"
                        print("loading testing results from :", test_path)
                        gt_path = f"/usa/dayu/CRS/RA-CRS/Re-rank/own_reranker/experiment_data/test_{generation_type}_{id_name}_{num_candidates}_{dataset_name}_{args.retriever}.json"
                        origin_data_dir = f"/usa/dayu/CRS/RA-CRS/Re-rank/own_reranker/data/fine-tune_data_{dataset_name}_{generation_type}_test_{num_candidates}_{args.retriever}.json"
                        origin_data = load_jsonl(origin_data_dir)  
                        #print(len(gt_path), len(origin_data), len(test_path))
                        
                        outputs = load_predictions(test_path)
                        if outputs is None:
                            match_ratio_percent = "N/A"  # Mark as N/A if the result file doesn't exist
                            halluciation_ratio_percent = "N/A"
                        else:
                            experiment_type = remove_underscores(generation_type) + '_' + id_name + '_' + str(num_candidates)
                            processed_data = process_dataset(gt_path, experiment_type)
                            gt_outputs = [data['generation'] for data in processed_data]
                            #experiment_type = f"{generation_type}_{id_name}_{num_candidates}"
                            assert len(gt_outputs) == len(origin_data) == len(processed_data)
                            match_ratio, halluciation_ratio = calculate_match_ratio(gt_outputs, outputs, generation_type, id_name, experiment_type, test_eval_setting, movie_table, num_candidates, args, origin_data)
                            match_ratio_percent = f"{match_ratio * 100:.2f}%"
                            halluciation_ratio_percent = f"{halluciation_ratio * 100:.2f}%"

                        results.append([dataset_name, generation_type, test_eval_setting, id_name, num_candidates, match_ratio_percent, halluciation_ratio_percent])
        
        print(f"Results for {dataset_name} dataset:")
        headers = ["Dataset", "Generation Type", "Test/Eval", "ID/Name", "Num Candidates", "Match Ratio (%)", "Hallucination Ratio (%)"]
        print(tabulate(results, headers=headers, tablefmt='pretty'))
        results = []  # Reset for next dataset

if __name__ == '__main__':
    main()
