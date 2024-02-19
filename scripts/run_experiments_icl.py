import os
from torch import cuda
from huggingface_hub import login, logout
import argparse
import getpass

from utils.pre_processing import gather_data
from utils.build_prompt import build_prompts
from utils.prediction_pipeline import run_pipeline, post_process_prediction, evaluate_predictions


def run_experiment_icl(args):
    
    dataset_id = args['dataset_id']
    model_id = args['model_id']
    task = args['task']
    N = args['N']
    batch_size = args['batch_size']
    padding_side = args['padding_side']
    debug = args['debug']
    seed = args['seed']  
    device = args['device']
    
    train_set, val_set, test_set = gather_data(
        dataset_id=dataset_id, 
        debug=debug
    )
        
    test_set['prompt'] = build_prompts(
        train_set=train_set, 
        test_set=test_set,
        N=N,
        seed=seed
    )
        
    test_set["model_output"] = run_pipeline(
        model=model_id, 
        task=task, 
        test_set=test_set, 
        batch_size=batch_size, 
        device=device,
        decoder_max_length=max(val_set["#_graph_tokens"]), 
        padding_side=padding_side
    ) 
        
    test_set["model_output_postprocessed"] = test_set["model_output"].map(lambda x: post_process_prediction(x))
        
    test_set = evaluate_predictions(
        test_set=test_set
    )
        
    test_set.to_csv('output/test_set_{}_{}_N-{}_batch_size-{}_seed-{}_debug-{}.csv'.format(model_id.replace("/", "_"), dataset_id, N, batch_size, seed, debug), index=False)
            
    print("Results saved")
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', type=str)
    parser.add_argument('--icl_examples', nargs='+', type=int, default=[8])
    parser.add_argument('--datasets', nargs='+', type=str, default=['web_nlg', 'bio_event'])
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--debug", type=bool, default=False)
    args = vars(parser.parse_args())
    
    assert args['models'] == ['all'] or args['models'] == ['mistral'] or args['models'] == ['llama'] or set(args['models']) <= set([
        'mistralai/Mistral-7B-v0.1', 
        'mistralai/Mistral-7B-Instruct-v0.1', 
        'Open-Orca/Mistral-7B-OpenOrca', 
        'meta-llama/Llama-2-7b-chat-hf', 
        'meta-llama/Llama-2-13b-chat-hf', 
        'epfl-llm/meditron-7b', 
    ]), "Please select a suitable list of models"
    
    assert set(args['icl_examples']) <= set([0, 2, 4, 8, 16, 32]), "Please select a suitable list of ICL examples"
    
    assert set(args['datasets']) <= set(['web_nlg', 'bio_event']), 'Please select a suitable list of datasets'
    
    hf_key = getpass.getpass(prompt='Please input Hugging Face key: ')
    login(hf_key) 
    
    args['device'] = 'cuda' if cuda.is_available() else 'cpu'
    print("Device: {}".format(args['device']))
    
    if args['models'] == ['all']:
        args['models'] = [
            'mistralai/Mistral-7B-v0.1', 
            'mistralai/Mistral-7B-Instruct-v0.1', 
            'Open-Orca/Mistral-7B-OpenOrca', 
            'meta-llama/Llama-2-7b-chat-hf', 
            'meta-llama/Llama-2-13b-chat-hf', 
            'epfl-llm/meditron-7b', 
        ]
    elif args['models'] == ['mistral']:
        args['models'] = [
            'mistralai/Mistral-7B-v0.1', 
            'mistralai/Mistral-7B-Instruct-v0.1', 
            'Open-Orca/Mistral-7B-OpenOrca', 
        ]
    elif args['models'] == ['llama']:
        args['models'] = [
            'meta-llama/Llama-2-7b-chat-hf', 
            'meta-llama/Llama-2-13b-chat-hf', 
            'epfl-llm/meditron-7b', 
        ]
    
    for model_id in args['models']:
        for dataset_id in args['datasets']:
            for N in args['icl_examples']:
                
                args['model_id'] = model_id
                args['dataset_id'] = dataset_id
                args['N'] = N
                args['task'] = 'text-generation'
                args['padding_side'] = 'left'
            
                print('Arguments: ', args)
            
                run_experiment_icl(args)
    
    logout()