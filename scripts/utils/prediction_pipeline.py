import torch
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
from datasets import Dataset
import evaluate


def build_pipeline(model, task, padding_side, device, tokenizer=None):
    print("Building HF pipeline object...")

    if task=="text2text-generation":
        pipe = pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            torch_dtype=torch.float16,
            use_fast=True,
            device=device,
        )
    elif task=='text-generation':
        pipe = pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            torch_dtype=torch.float16,
            use_fast=True,
            device=device,
            return_full_text=False,
        )

    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    pipe.tokenizer.padding_side = padding_side
        
    print("Pipeline object built")
        
    return pipe


def run_pipeline(model, task, test_set, batch_size, decoder_max_length, padding_side, device, tokenizer=None):
    
    print("Running experiments...")
    
    pipe = build_pipeline(
        model=model,
        task=task,
        device=device,
        padding_side=padding_side,
        tokenizer=tokenizer,
    )
    
    experiment_results = []
    
    keydataset = Dataset.from_pandas(test_set)
    
    for output in tqdm(pipe(
        KeyDataset(keydataset, "prompt"),
        batch_size=batch_size,
        max_new_tokens=decoder_max_length,
        num_return_sequences=1,
        pad_token_id=pipe.tokenizer.eos_token_id,
        eos_token_id=pipe.tokenizer.eos_token_id,
    )):
        experiment_results.append(output[0]['generated_text'])
            
    print("Experiments ran")
    
    return experiment_results


def post_process_prediction(model_output):
    
    model_output.split(')]')[0] + ')]' if ')]' in model_output else model_output
    
    model_output.lstrip()

    return model_output


def evaluate_predictions(test_set):
    
    print("Evaluating predictions...")
    
    model_output = test_set["model_output"]
    references = test_set["graph"]
    
    rouge = evaluate.load('rouge')
    
    rouge_scores = rouge.compute(
            predictions=model_output,
            references=references,
            use_aggregator=False
    )
    
    test_set = test_set.assign(
        rouge_1 = rouge_scores['rouge1'], 
        rouge_2 = rouge_scores['rouge2'],
        rouge_l = rouge_scores['rougeL'],
    )
    
    if "model_output_postprocessed" in test_set.columns:
        model_output_postprocessed = test_set['model_output_postprocessed']
            
        rouge_scores_postprocessed = rouge.compute(
            predictions=model_output_postprocessed,
            references=references,
            use_aggregator=False
        )
        
        test_set = test_set.assign(
            rouge_1_postprocessed = rouge_scores_postprocessed['rouge1'], 
            rouge_2_postprocessed = rouge_scores_postprocessed['rouge2'],
            rouge_l_postprocessed = rouge_scores_postprocessed['rougeL'],
        )
        
    print("Predictions evaluated")
        
    return test_set