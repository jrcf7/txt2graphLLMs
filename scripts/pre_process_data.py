import pandas as pd
from datasets import load_dataset

from utils.pre_processing import bio_event_process_dataset, bio_event_filter_dataset, bio_event_save_dataset, web_nlg_process_dataset, web_nlg_save_dataset


if __name__== '__main__':
    
    print("Gathering data for Bio Event...")
    
    train_set = pd.read_csv("data/Bio_Event/original/bio_event_e2t_{}.tsv".format("train"), sep='\t')
    val_set = pd.read_csv("data/Bio_Event/original/bio_event_e2t_{}.tsv".format("dev"), sep='\t')
    test_set = pd.read_csv("data/Bio_Event/original/bio_event_e2t_{}.tsv".format("test"), sep='\t')
    dataset = pd.concat([train_set, val_set, test_set])
    
    print("Data gathered")
    print("Train size: {}; Val size: {}; Test size: {}".format(
        len(train_set),
        len(val_set),    
        len(test_set),
    ))
    
    dataset_processed = bio_event_process_dataset(dataset)
    dataset_filtered = bio_event_filter_dataset(dataset_processed)
    bio_event_save_dataset(dataset_filtered)
    
    print("Gathering data for Web NLG...")
    
    dataset = load_dataset("web_nlg", "release_v3.0_en")
    train_set = dataset['train'].to_pandas()
    val_set = dataset['dev'].to_pandas()
    test_set = dataset['test'].to_pandas()
    test_set = test_set[test_set['test_category'] == 'rdf-to-text-generation-test-data-with-refs-en']
    
    train_set.to_csv("data/Web_NLG/original/web_nlg_release_v3.0_en_{}.csv".format("train"), index=False)
    val_set.to_csv("data/Web_NLG/original/web_nlg_release_v3.0_en_{}.csv".format("dev"), index=False)
    test_set.to_csv("data/Web_NLG/original/web_nlg_release_v3.0_en_{}.csv".format("test"), index=False)
    
    print("Data gathered")
    
    print("Train size: {}; Val size: {}; Test size: {}".format(
        len(train_set),
        len(val_set),    
        len(test_set),
    ))
    
    train_set_processed = web_nlg_process_dataset(train_set)
    val_set_processed = web_nlg_process_dataset(val_set)
    test_set_processed = web_nlg_process_dataset(test_set)
    web_nlg_save_dataset(train_set_processed, val_set_processed, test_set_processed)
    