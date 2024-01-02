import pandas as pd

def gather_sources(dataset_id, debug=False, seed=41):
    
    datasets = {
        "web_nlg" : "data/Web_NLG/web_nlg_release_v3.0_en_{}.csv",
        "bio_event" : "data/BioEv/e2t/BioEv_e2t_{}.csv",
    }
        
    print("Gathering data for ", dataset_id, "...")
    
    if dataset_id == "web_nlg":
        train_set = pd.read_csv(datasets[dataset_id].format("train"))
        val_set = pd.read_csv(datasets[dataset_id].format("dev"))
        test_set = pd.read_csv(datasets[dataset_id].format("test"))

    elif dataset_id == "bio_event":
        train_set = pd.read_csv(datasets[dataset_id].format("train"))
        test_set = pd.read_csv(datasets[dataset_id].format("test"))
        val_set = pd.read_csv(datasets[dataset_id].format("dev"))
            
    if debug:
        print("!!! Using debug mode !!!")
        
        train_set = train_set.sample(frac=.001, random_state=seed).reset_index(drop=True)
        val_set = val_set.sample(frac=.001, random_state=seed).reset_index(drop=True)
        test_set = test_set.sample(frac=.001, random_state=seed).reset_index(drop=True)
            
    print("Data gathered")
    
    print("Train size: {}; val size: {}; test size: {}".format(
        len(train_set),
        len(val_set),    
        len(test_set),
    ))
        
    return(train_set, val_set, test_set)