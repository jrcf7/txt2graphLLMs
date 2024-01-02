import os
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from huggingface_hub import login
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
import argparse

from utils.gather_sources import gather_sources
from utils.post_processing import post_process_predictions
from utils.evaluate_predictions import evaluate_predictions



os.environ["HF_HOME"] = "/home/diglifeproc/DIGLIFE-scratch/cache/huggingface/"

hf_login_key = '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'

login(hf_login_key)


class pipeline_ft:
    """
        HF pipeline for FT of LLMs
    """
    
    def __init__(self, args):
        self.args = args
    
    
    def build_hf_objects(self, model_id):
        
        print("Initiating tokenizer/model/datacollator...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        
        tokenizer.pad_token = tokenizer.eos_token
        
        print("Tokenizer/model/datacollator initiated")
        
        return tokenizer, model, data_collator
    
    
    def __build_ft_queries_helper(self, data_set, eos_token):
        
        ft_queries = []
        
        for text, _, triplets, _, _ in tqdm(data_set.values):
            
            ft_prompt = "Convert the text into a sequence of triplets:\n"
            ft_prompt += "Text: {}\nGraph: {}".format(text, triplets)
            ft_prompt += eos_token
            
            ft_queries.append(ft_prompt)
            
        return ft_queries
    
    
    def build_ft_queries(self, train_set, val_set, eos_token):
        
        print("Building train and validation prompts...")
            
        train_ft_prompts = self.__build_ft_queries_helper(
            data_set=train_set, 
            eos_token=eos_token)
            
        val_ft_prompts = self.__build_ft_queries_helper(
            data_set=val_set, 
            eos_token=eos_token)
        
        print("Train and validation prompts built")
        
        return train_ft_prompts, val_ft_prompts
    
    
    def tokenize_ft_queries(self, train_set, val_set, tokenizer, model):
        
        print("Tokenizing prompts...")
        train_ft_prompts_tokenized = tokenizer(
            list(train_set["ft_prompts"].values),
            truncation=False,
            max_length=model.config.max_position_embeddings,
            return_overflowing_tokens=False,
        )["input_ids"]
        
        val_ft_prompts_tokenized = tokenizer(
            list(val_set["ft_prompts"].values),
            truncation=False,
            max_length=model.config.max_position_embeddings,
            return_overflowing_tokens=False,
        )["input_ids"]
        print("Prompts tokenized")
        
        return train_ft_prompts_tokenized, val_ft_prompts_tokenized
    
    
    def run_experiments_ft(self, train_set, val_set, tokenizer, model, data_collator):
        print("Running finetuning...")
        
        batch_size = self.args['batch_size']
        
        print("Initiating trainer...")
        args = TrainingArguments(
            output_dir="output/ft",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps",
            eval_steps=5_000,
            logging_steps=5_000,
            num_train_epochs=1,
            weight_decay=0.1,
            warmup_steps=1_000,
            lr_scheduler_type="cosine",
            learning_rate=5e-4,
            save_steps=5_000,
            fp16=True,
            push_to_hub=False,
        )
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=train_set['ft_prompts_tokenized'],
            eval_dataset=val_set['ft_prompts_tokenized'],
        )
        print("Trainer initiated")
        
        print("Starting finetuning...")
        trainer.train()
        print("Finetuning finished")
        
        return trainer
    
    
    def save_results(self, train_set, val_set):
        print("Saving results...")
        
        dataset_id = self.args['dataset_id']
        model_id = self.args['model_id']
        batch_size = self.args['batch_size']
        seed = self.args['seed']
        debug = self.args['debug']
        
        train_set.to_csv('output/ft/train_set_{}_{}_batch_size-{}_seed-{}_debug-{}.csv'.format(dataset_id, model_id.replace("/", "_"), batch_size, seed, debug))
        
        val_set.to_csv('output/ft/val_set_{}_{}_batch_size-{}_seed-{}_debug-{}.csv'.format(dataset_id, model_id.replace("/", "_"), batch_size, seed, debug))
        
        print("Data saved")
    
        
    def main(self):
        print("Running code...")
        
        dataset_id = self.args['dataset_id']
        model_id = self.args['model_id']
        debug = self.args['debug']
        seed = self.args['seed']
        
        train_set, val_set, test_set = gather_sources(
            dataset_id=dataset_id, 
            debug=debug, 
            seed=seed
        )
        
        tokenizer, model, data_collator = self.build_hf_objects(
            model_id=model_id
        )
        
        train_set['ft_prompts'], val_set['ft_prompts'] = self.build_ft_queries(
            train_set=train_set, 
            val_set=val_set, 
            eos_token=tokenizer.eos_token
        )
        
        train_set['ft_prompts_tokenized'], val_set['ft_prompts_tokenized'] = self.tokenize_ft_queries(
            train_set=train_set,
            val_set=val_set,
            tokenizer=tokenizer,
            model=model
        )
        
        trainer = self.run_experiments_ft(
            train_set=train_set, 
            val_set=val_set, 
            tokenizer=tokenizer, 
            model=model, 
            data_collator=data_collator
        ) 
        
        self.save_results(
            train_set=train_set, 
            val_set=val_set
        )
        
        print("Code finished")
        
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', type=str)
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
    
    
    assert set(args['datasets']) <= set(['web_nlg', 'bio_event']), 'Please select a suitable list of datasets, i.e. as web_nlg and/or bio_event'

    for model_id in args['models']:
        for dataset_id in args['datasets']:
            args['model_id'] = model_id
            args['dataset_id'] = dataset_id
                
            print('Arguments: ', args)
            
            pipeline_ft = pipeline_ft(args)
            
            pipeline_ft.main()