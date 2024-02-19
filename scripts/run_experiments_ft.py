import os
from torch import cuda
from huggingface_hub import login, logout
import argparse
import getpass
from transformers import Seq2SeqTrainingArguments

from utils.pre_processing import gather_data
from utils.fine_tuning import trainer_class
from utils.prediction_pipeline import run_pipeline, evaluate_predictions


def run_experiment_ft(args):
        
    model_id = args['model_id']
    dataset_id = args['dataset_id']
    task = args['task']
    padding_side = args['padding_side']
    batch_size = args['batch_size']
    device = args['device']
    epochs = args['epochs']
    debug = args['debug']
    seed = args['seed']
    hf_name = args['hf_name']
    
    train_set, val_set, test_set = gather_data(
        dataset_id=dataset_id, 
        debug=debug
    )
    
    training_args = Seq2SeqTrainingArguments(
        seed=seed,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_rouge1",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        label_smoothing_factor=0.1,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        generation_max_length=max(val_set["#_graph_tokens"]),
        fp16=True,
        optim="adamw_torch",
        load_best_model_at_end=True,
        warmup_steps=10,
        logging_steps=10,
        group_by_length=True,
        length_column_name="#_graph_tokens",
        push_to_hub=True if debug==False else False,
        hub_private_repo=True,
        hub_strategy='all_checkpoints',
        hub_model_id="{}/{}_{}_N-0_batch_size-{}_seed-{}_debug-{}".format(hf_name, model_id.replace("/", "_"), dataset_id, batch_size, seed, debug),
        output_dir="output/{}_{}_N-0_batch_size-{}_seed-{}_debug-{}".format(model_id.replace("/", "_"), dataset_id, batch_size, seed, debug),
    )

    print("Starting fine-tuning...")
    
    trainer = trainer_class(
        model_id=model_id,
        device=device
    ).initiate_trainer(
        train_set=train_set, 
        val_set=val_set, 
        training_args=training_args
    )
    
    trainer.train()
    
    print("End of fine-tuning")
    
    trainer.save_model()
    
    print("Model saved")

    print("Running prediction pipeline...")
    
    test_set['prompt'] = test_set['text']
    
    test_set["model_output"] = run_pipeline(
        model=trainer.model, 
        tokenizer=trainer.tokenizer,
        device=device,
        task=task, 
        test_set=test_set, 
        batch_size=batch_size, 
        decoder_max_length=max(val_set["#_graph_tokens"]), 
        padding_side=padding_side
    ) 
    
    test_set = evaluate_predictions(
        test_set=test_set
    )
    
    test_set.to_csv('output/test_set_{}_{}_N-0_batch_size-{}_seed-{}_debug-{}.csv'.format(model_id.replace("/", "_"), dataset_id, batch_size, seed, debug), index=False)
            
    print("Results saved")

    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', type=str)
    parser.add_argument('--datasets', nargs='+', type=str, default=['web_nlg', 'bio_event'])
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--debug", type=bool, default=False)
    args = vars(parser.parse_args())
    
    assert args['models'] == ['all'] or set(args['models']) <= set([
        't5-small',
        't5-base',
        't5-large',
        'facebook/bart-large',
        'ibm/knowgl-large',
    ]), "Please select a suitable list of models"
    
    assert set(args['datasets']) <= set(['web_nlg', 'bio_event']), 'Please select a suitable list of datasets'

    args['hf_name'] = input("Please input Hugging Face username: ")
    hf_key = getpass.getpass(prompt='Please input Hugging Face key: ')
    
    login(hf_key) 
    
    args['device'] = 'cuda' if cuda.is_available() else 'cpu'
    print("Device: {}".format(args['device']))
    
    if args['models'] == ['all']:
        args['models'] = [
            't5-small',
            't5-base',
            't5-large',
            'facebook/bart-large',
            'ibm/knowgl-large',
        ]
    
    for model_id in args['models']:
        for dataset_id in args['datasets']:
                
            args['model_id'] = model_id
            args['dataset_id'] = dataset_id
            args['task'] = "text2text-generation"
            args['padding_side'] = 'right'
            
            print('Arguments: ', args)
            
            run_experiment_ft(args)
    
    logout()


