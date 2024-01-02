import os
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from huggingface_hub import login
import torch
from tqdm import tqdm
from datasets import Dataset
import pickle
import argparse

from utils.gather_sources import gather_sources
from utils.post_processing import post_process_predictions
from utils.evaluate_predictions import evaluate_predictions


hf_login_key = LOGIN_KEY

login(hf_login_key)


class pipeline_icl:
    """
        HF pipeline for ICL for LLMs
    """
    
    def __init__(self, args):
        self.args = args

        
    def build_icl_prompt(self, train_set, seed):
        
        N = self.args['N']
        
        # Select the labelled examples for iCL
        icl_examples = train_set.sample(N, random_state=seed)
            
        icl_prompt = "Convert the text into a sequence of triplets:\n"
        for icl_text, _, icl_triplets, _, _ in icl_examples.values:
            icl_prompt += "Text: {}\nGraph: {}\n".format(icl_text, icl_triplets)
            
        return icl_prompt
        
        
    def build_icl_queries(self, train_set, test_set, seed):
        
        print("Building ICL queries...")
        
        icl_queries = []
        
        for text, _, triplets, _, _ in tqdm(test_set.values):
            seed += 1
            icl_prompt = self.build_icl_prompt(
                train_set=train_set, 
                seed=seed
            )
            
            icl_prompt += f"Text: {text}\nGraph:"
            icl_queries.append(icl_prompt)
        
        print("Test set built")
        
        return icl_queries
        
    
    def __build_pipe(self):
        print("Building HF pipeline object...")
        
        # Set up the pipeline object
        model_id = self.args['model_id']
        task = self.args['task']
        padding_side = self.args['padding_side']

        pipe = pipeline(
            task=task,
            model=model_id,
            do_sample=False,
            torch_dtype=torch.float16,
            device_map="auto",
            return_full_text=False,
            use_fast=True,
        )
        pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
        pipe.tokenizer.padding_side = padding_side
        
        print("Pipeline object built")
        
        return pipe
    
    
    def __compute_max_length(self, train_set, val_set):
        
        print("Calculating max encoder/decoder length...")
        
        encoder_max_length = max(train_set["#_text_tokens"]) 
        decoder_max_length = max(val_set["#_triplets_tokens"])
                
        print("Encoder max length: {}; Decoder max length: {}".format(
            encoder_max_length, 
            decoder_max_length
        ))
        
        return(encoder_max_length, decoder_max_length)

    
    def run_experiments(self, train_set, val_set, test_set):
        
        print("Running experiments...")
        
        batch_size = self.args['batch_size']
        
        encoder_max_length, decoder_max_length = self.__compute_max_length(
            train_set=train_set,
            val_set=val_set,
        )
        pipe = self.__build_pipe()
        
        experiment_results = []
        
        keydataset = Dataset.from_pandas(test_set)
        
        for out in tqdm(pipe(
            KeyDataset(keydataset, "icl_prompts"),
            batch_size=batch_size,
            max_new_tokens=decoder_max_length,
            num_return_sequences=1,
            pad_token_id=pipe.tokenizer.eos_token_id,
            eos_token_id=pipe.tokenizer.eos_token_id,
        )):
            experiment_results.append(out[0]['generated_text'])
        
        print("Experiments ran")
        
        return experiment_results

    
    def save_results(self, rouge_scores, rouge_scores_trunc, test_set):
        print("Saving results...")
        
        dataset_id = self.args['dataset_id']
        model_id = self.args['model_id']
        N = self.args['N']
        batch_size = self.args['batch_size']
        seed = self.args['seed']
        debug = self.args['debug']
        
        with open('output/icl/rouge_{}_{}_N-{}_batch_size-{}_seed-{}_debug-{}.pickle'.format(dataset_id, model_id.replace("/", "_"), N, batch_size, seed, debug), 'wb') as handle:
            pickle.dump(rouge_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open('output/icl/rouge_trunc_{}_{}_N-{}_batch_size-{}_seed-{}_debug-{}.pickle'.format(dataset_id, model_id.replace("/", "_"), N, batch_size, seed, debug), 'wb') as handle:
            pickle.dump(rouge_scores_trunc, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        test_set.to_csv('output/icl/test_set_{}_{}_N-{}_batch_size-{}_seed-{}_debug-{}.csv'.format(dataset_id, model_id.replace("/", "_"), N, batch_size, seed, debug))
            
        print("Results saved")
        
        
        
    def main(self):
        print("Running code...")
        
        dataset_id = self.args['dataset_id']
        debug = self.args['debug']
        seed = self.args['seed']
        
        train_set, val_set, test_set = gather_sources(
            dataset_id=dataset_id, 
            debug=debug, 
            seed=seed
        )
        
        test_set['icl_prompts'] = self.build_icl_queries(
            train_set=train_set, 
            test_set=test_set,
            seed=seed
        )
        
        test_set["icl_predictions"] = self.run_experiments(
            train_set=train_set, 
            val_set=val_set, 
            test_set=test_set
        ) 
        
        test_set['icl_predictions_trunc'] = post_process_predictions(
            predictions=test_set["icl_predictions"]
        )
            
        rouge_scores, rouge_scores_trunc, test_set = evaluate_predictions(
            test_set=test_set
        )
        
        self.save_results(
            rouge_scores=rouge_scores,
            rouge_scores_trunc=rouge_scores_trunc, 
            test_set=test_set
        )
        
        print("Code finished")
        
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', type=str)
    parser.add_argument('--icl_examples', nargs='+', type=int)
    parser.add_argument('--datasets', nargs='+', type=str, default=['web_nlg', 'bio_event'])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--debug", type=bool, default=False)
    args = vars(parser.parse_args())
    
    assert set(args['models']) <= set([
        'mistralai/Mistral-7B-v0.1', 
        'mistralai/Mistral-7B-Instruct-v0.1', 
        'Open-Orca/Mistral-7B-OpenOrca', 
        'epfl-llm/meditron-7b', 
        'meta-llama/Llama-2-7b-chat-hf', 
        'meta-llama/Llama-2-13b-chat-hf', 
        't5-small',
        't5-base',
        't5-large',
        't5-3b',
        't5-11b',
        'facebook/bart-large',
        'Babelscape/rebel-large',
    ]), "Please select a suitable list of models"
    
    assert set(args['icl_examples']) <= set([0, 2, 4, 8, 16, 32, 64]), "Please select a suitable list of ICL examples, i.e. as a subset of [0, 2, 4, 8, 16, 32, 64]"
    
    assert set(args['datasets']) <= set(['web_nlg', 'bio_event']), 'Please select a suitable list of datasets, i.e. as web_nlg and/or bio_event'
    

    for model_id in args['models']:
        for dataset_id in args['datasets']:
            for N in args['icl_examples']:
                args['model_id'] = model_id
                args['dataset_id'] = dataset_id
                args['N'] = N
                
                if model_id in [
                    'mistralai/Mistral-7B-v0.1', 
                    'mistralai/Mistral-7B-Instruct-v0.1', 
                    'Open-Orca/Mistral-7B-OpenOrca', 
                    'meta-llama/Llama-2-7b-chat-hf', 
                    'meta-llama/Llama-2-13b-chat-hf',
                    'epfl-llm/meditron-7b',
                ]:
                    args['task'] = 'text-generation'
                    args['padding_side'] = 'left'
                elif model_id in [
                    't5-small',
                    't5-base',
                    't5-large',
                    't5-3b',
                    't5-11b'
                ]:
                    args['task'] = 'translation'
                    args['padding_side'] = 'right'
                elif model_id in ['facebook/bart-large']:
                    args['task'] = 'feature-extraction'
                    args['padding_side'] = 'right'
                elif model_id in ['Babelscape/rebel-large']:
                    args['task'] = 'text2text-generation'
                    args['padding_side'] = 'right'
            
                print('Arguments: ', args)
            
                pipeline_icl = pipeline_icl(args)
            
                pipeline_icl.main()
