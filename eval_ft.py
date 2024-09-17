import re
import argparse
from tqdm import tqdm
import copy
import datasets
import json
from fuzzywuzzy import fuzz
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig, AutoTokenizer
import safetensors.torch
from peft import PeftModel
import torch
from accelerate.utils import is_xpu_available
import pandas as pd
import os



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

IGNORE_INDEX = -100




def create_movie_search_table(file_path, column_name='primaryTitle'):
    # Create an empty set to store unique movie names in lowercase
    movie_names_set = set()

    # Read the TSV file in chunks
    chunk_size = 5000  # Adjust chunk size based on your system's memory capacity
    for chunk in pd.read_csv(file_path, sep='\t', usecols=[column_name], chunksize=chunk_size):
        # Update the set with movie names from the current chunk, transformed to lowercase
        movie_names_set.update(chunk[column_name].str.lower().dropna().unique())

    return movie_names_set

def is_movie_in_database(movie_name, search_table):
    # Check if the movie name (in lowercase) is in the search table
    return movie_name.lower() in search_table



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





def inference(model_path, adapter_path, input_texts):
    '''
    This function performs inference on a list of input texts using a fine-tuned LoRA large language model.
    
    Parameters:
    - model_path: str, path to the pre-trained model directory
    - adapter_path: str, path to the adapter model
    - input_texts: list of str, a list of input texts for generation
    
    Returns:
    - list of str: generated texts corresponding to each input text
    '''

    max_new_tokens = 250
    do_sample = True
    top_p = 0.9
    temperature = 0.05
    min_length = None

    def load_model(model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        return model


    # Function to load the PeftModel for performance optimization
    def load_peft_model(model, peft_model):
        peft_model = PeftModel.from_pretrained(model, peft_model)
        return peft_model


    def run_inference(model, input_texts, llama_model_dir, batch_size=1):
        # Only support batch_size = 1!
        tokenizer = AutoTokenizer.from_pretrained(llama_model_dir)
        tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.eos_token_id

        output_texts = []
        # Process inputs in batches
        for i in tqdm(range(0, len(input_texts), batch_size)):
            batch_texts = input_texts[i:i + batch_size]
            batch = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=4096, return_tensors="pt")

            # Determine device type
            # if is_xpu_available():
            #     batch = {k: v.to("xpu") for k, v in batch.items()}
            # else:
            #     batch = {k: v.to("cuda") for k, v in batch.items()}

            if is_xpu_available():
                batch = {k: v.to("xpu") for k, v in batch.items()}
            else:
                batch = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in batch.items()}


            with torch.no_grad():
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    pad_token_id=pad_token_id

                )

            # Decode each output into text and add to results
            batch_output_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            output_texts.extend(batch_output_texts)

        return output_texts
    
    

    # Load the model and set to evaluation mode
    model = load_model(model_path)
    model = load_peft_model(model, adapter_path)
    model.eval()

    # Run inference on the batch of input texts
    output_texts = run_inference(model, input_texts, model_path)
    
    return output_texts

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



def is_match_high(string1, string2, threshold=90):
    # Calculate the fuzzy match ratio
    match_ratio = fuzz.ratio(string1, string2)
    # Compare the match ratio with the threshold
    return match_ratio > threshold

def extract_digit_between_identifiers(long_string):
    '''
    return : a list of string of digits (didn't covnert to int)
    
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



def clean_movie_titles(title):
    # Regular expression to find a four-digit year within parentheses
    cleaned_title = re.sub(r'\s*\(\d{4}\)', '', title).strip()
    return cleaned_title

def calculate_match_ratio(gt_outputs, outputs, generation_type, id_or_name, experiment_type, test_or_eval, search_table, num_candidates):
    '''
    
    return: a list of 0 or 1, 1 means match, 0 means not match
        - length of list is the same as the length of gt_outputs
    '''
    experiment_type = remove_after_last_underscore(experiment_type)
    match_ratio = []
    error = 0
    halluciation = 0
    for gt_output, output in zip(gt_outputs, outputs):
        # a single gt_output is a list of strings
        predict = extract_after_marker(output, MARKERS[experiment_type])
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
                    if int(gt) == int(predict):
                        success = 1
                        break
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
    print("\n" + "#"*30)
    print(f"Calculate match; Error ratio for {generation_type}, {experiment_type}, {test_or_eval} :", error/len(outputs))
    print(f"Halluciation for {generation_type}, {experiment_type}, {test_or_eval} :", halluciation/(len(outputs)-error))
    return match_ratio







def store_predictions(outputs, file_path, experiment_type):
    predictions = outputs #[extract_after_marker(output, MARKERS[experiment_type]) for output in outputs]
    with open(file_path, 'w') as f:
        json.dump(predictions, f)
        
def load_or_infer(file_path, model_path, adapter_path, input_texts):
    if os.path.exists(file_path):
        print(f"Loading predictions from {file_path}")
        with open(file_path, 'r') as f:
            predictions = json.load(f)
    else:
        print("Performing inference...")
        predictions = inference(model_path, adapter_path, input_texts)
        with open(file_path, 'w') as f:
            json.dump(predictions, f)
    return predictions


def parsing_args():
    parser = argparse.ArgumentParser(description='Eval ft llama on eval dataset')
    parser.add_argument('--generation_type', type=str, default='no_reason', choices=['no_reason','reason'])
    parser.add_argument('--id_or_name', type=str, default='name', choices=['id', 'name'])
    parser.add_argument('--test_or_eval', type=str, default="test", choices = ['test', 'eval'], help="eval = on the split-test set(same distribution as training), test = actual testing set (inspired and redial)")
    parser.add_argument('--num_candidates', type=int, default=50, help="num of candidate items from retrieval stage")
    parser.add_argument('--test_dataset', type=str, default='inspired', choices=['inspired', 'redial'], help='test dataset to evaluate on')
    parser.add_argument('--retriever', type=str, default='bge15', choices=['bge15', 'openai', 'sfr'], help='Retriever to use at retriever stage')
    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = parsing_args()
    
    # Set file paths based on test or eval mode
    if args.test_or_eval == 'eval':
        file_dir = f"/usa/dayu/CRS/RA-CRS/Re-rank/own_reranker/ft_data/fine-tune_data_eval_{args.generation_type}_{args.id_or_name}_{args.num_candidates}.json"
    elif args.test_or_eval == 'test':
        inspired_test = f"/usa/dayu/CRS/RA-CRS/Re-rank/own_reranker/experiment_data/test_{args.generation_type}_{args.id_or_name}_{args.num_candidates}_inspired_{args.retriever}.json"
        redial_test = f"/usa/dayu/CRS/RA-CRS/Re-rank/own_reranker/experiment_data/test_{args.generation_type}_{args.id_or_name}_{args.num_candidates}_redial_{args.retriever}.json"
    else:
        raise ValueError("Not implemented yet")
    
    # build movie search table
    imdb_tsv_dir = "/data_hdd/dayu/RALLM_CRS/hallucination/names.tsv"
    movie_table = create_movie_search_table(imdb_tsv_dir)
    
    # inference save path
    eval_path = f"/usa/dayu/CRS/RA-CRS/Re-rank/own_reranker/test_results/eval_{args.generation_type}_{args.id_or_name}_{args.test_or_eval}_{args.num_candidates}_{args.retriever}.json"
    inspired_test_path = f"/usa/dayu/CRS/RA-CRS/Re-rank/own_reranker/test_results/test_{args.generation_type}_{args.id_or_name}_{args.test_or_eval}_{args.num_candidates}_inspired_{args.retriever}.json"
    redial_test_path = f"/usa/dayu/CRS/RA-CRS/Re-rank/own_reranker/test_results/test_{args.generation_type}_{args.id_or_name}_{args.test_or_eval}_{args.num_candidates}_redial_{args.retriever}.json"
    
    
    # Set model and adapter paths
    model_path = '/data_hdd/dayu/RALLM_CRS/Meta-Llama-3-8B-Instruct-hf'
    adapter_path = f'/data_hdd/dayu/RALLM_CRS/llama3_ft/{remove_underscores(args.generation_type)}_{args.id_or_name}_{args.num_candidates}/'
    
    # Set experiment type
    experiment_type = remove_underscores(args.generation_type) + '_' + args.id_or_name + '_' + str(args.num_candidates)

    if args.test_or_eval == 'eval':
        print(" Under eval mode, similar to dev set, not actual testing results")
        processed_data = process_dataset(file_dir, experiment_type)
        input_texts = [data['prompt'] for data in processed_data]
        gt_outputs = [data['generation'] for data in processed_data]
        
        outputs = load_or_infer(eval_path, model_path, adapter_path, input_texts)
        match_ratio = calculate_match_ratio(gt_outputs, outputs, args.generation_type, args.id_or_name, experiment_type, args.test_or_eval, movie_table, args.num_candidates)
        print("Match ratio: ", sum(match_ratio) / len(match_ratio))
    
    else:  # Test mode
        print(" Under test mode,  actual testing results")
        if args.test_dataset == 'inspired':
            processed_inspired = process_dataset(inspired_test, experiment_type)
            input_texts = [data['prompt'] for data in processed_inspired]
            gt_outputs = [data['generation'] for data in processed_inspired] # list of list
            outputs = load_or_infer(inspired_test_path, model_path, adapter_path, input_texts)
            match_ratio = calculate_match_ratio(gt_outputs, outputs, args.generation_type, args.id_or_name,experiment_type, args.test_or_eval, movie_table, args.num_candidates)
            print("Using retriever ", args.retriever, " for inspired dataset")
            print("Inspired match ratio: ", sum(match_ratio) / len(match_ratio))
        elif args.test_dataset == 'redial':
            processed_redial = process_dataset(redial_test, experiment_type)
            input_texts = [data['prompt'] for data in processed_redial]
            gt_outputs = [data['generation'] for data in processed_redial]
            outputs = load_or_infer(redial_test_path, model_path, adapter_path, input_texts)
            match_ratio = calculate_match_ratio(gt_outputs, outputs, args.generation_type, args.id_or_name,experiment_type, args.test_or_eval, movie_table, args.num_candidates)
            print("Using retriever ", args.retriever, " for redial dataset")
            print("Redial match ratio: ", sum(match_ratio) / len(match_ratio))
        else:
            raise ValueError("Not implemented yet")
    
    
    
    
    
if __name__ == '__main__':
    main()