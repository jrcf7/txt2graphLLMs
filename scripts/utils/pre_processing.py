import pandas as pd
import nltk

def gather_data(dataset_id, debug=False):
    
    datasets = {
        "web_nlg" : "data/Web_NLG/processed/web_nlg_release_v3.0_en_{}.csv",
        "bio_event" : "data/Bio_Event/processed/bio_event_e2t_{}.csv",
    }
    
    print("Gathering data for ", dataset_id, "...")
    
    train_set = pd.read_csv(datasets[dataset_id].format("train"))
    val_set = pd.read_csv(datasets[dataset_id].format("dev"))
    test_set = pd.read_csv(datasets[dataset_id].format("test"))
            
    if debug:
        print("!!! Using debug mode !!!")
        train_set = train_set.sample(frac=.001, random_state=41).reset_index(drop=True)
        val_set = val_set.sample(frac=.001, random_state=41).reset_index(drop=True)
        test_set = test_set.sample(frac=.001, random_state=41).reset_index(drop=True)
            
    print("Data gathered")
    
    print("Train size: {}; Val size: {}; Test size: {}".format(
        len(train_set),
        len(val_set),    
        len(test_set),
    ))
        
    return(train_set, val_set, test_set)


def web_nlg_process_text(text):
    nr_texts = len(text['lid'])
    if nr_texts>0:
        text_processed = text['text'][0]
    else:
        text_processed = ''
        
    return text_processed


def web_nlg_process_graph(graph):
    triplets = graph['mtriple_set'][0].tolist()
    triplets = [triplet.replace('|', '#') for triplet in triplets]
    graph_processed = ') | ('.join(triplets)
    graph_processed = '[(' + graph_processed + ')]'
    graph_processed = graph_processed.replace('\'', '')
    graph_processed = graph_processed.replace('\"', '')
    graph_processed = graph_processed.replace('_', ' ')
    
    return graph_processed


def web_nlg_process_dataset(dataset):
    dataset_processed = pd.DataFrame()
    dataset_processed['category'] = dataset['category']
    dataset_processed['text'] = dataset['lex'].map(
        lambda x: web_nlg_process_text(x))
    dataset_processed['graph'] = dataset['modified_triple_sets'].map(
        lambda x: web_nlg_process_graph(x))
    dataset_processed['#_triplets'] = dataset_processed['graph'].map(
        lambda x: len(x.split("|"))
    )
    dataset_processed['#_text_tokens'] = dataset_processed['text'].map(
        lambda x: len(nltk.word_tokenize(x))
    )
    dataset_processed['#_graph_tokens'] = dataset_processed['graph'].map(
        lambda x: len(nltk.word_tokenize(x))
    )

    print("Data processed")
    
    return dataset_processed


def web_nlg_save_dataset(train_set, val_set, test_set):
    train_set.to_csv('data/Web_NLG/processed/web_nlg_release_v3.0_en_train.csv', index=False)
    val_set.to_csv('data/Web_NLG/processed/web_nlg_release_v3.0_en_dev.csv', index=False)
    test_set.to_csv('data/Web_NLG/processed/web_nlg_release_v3.0_en_test.csv', index=False)
    
    print("Train size: {}; Val size: {}; Test size: {}".format(
        len(train_set),
        len(val_set),    
        len(test_set),
    ))
    
    print("Data saved")


def bio_event_process_text(text):
    
    if "', '" in text: # then mention is a list...
        text_processed = "".join(text[2:-2].split("', '"))
    else: # mention is a ctually a string!
        text_processed = text.replace("[", "").replace("]", "")
        
    return text_processed


def bio_event_process_graph(graph):
    
    if (graph == "ND") or (graph.count("|") == 1):
        graph_processed = "[]"
    else:
        graph_triplets = graph.replace("[[", "[").replace("]]", "]")
        graph_triplets = graph_triplets.replace("][", "]<>[").split("<>")
        graph_triplets = graph_triplets[1:] # the first has central-node info
    
        graph_processed = "["
        for c, triplet in enumerate(graph_triplets):
            triplet_elements = triplet[1:-1].split(" | ") # remove [ and ] and split
            t = triplet_elements[0]
            r, h = triplet_elements[-1].split(" = ") # spelled e.g., Theme = promote
            
            graph_processed += "({} # {} # {})".format(h,r,t)
            if c+1 == len(graph_triplets):
                graph_processed += "]"
            else:
                graph_processed += " | "
                
    return graph_processed


def bio_event_process_dataset(dataset):
    
    dataset_processed = pd.DataFrame()
    dataset_processed['text'] = dataset['event_mention'].map(
        lambda x: bio_event_process_text(x)
    )
    dataset_processed['graph'] = dataset['event'].map(
        lambda x: bio_event_process_graph(x)
    )
    dataset_processed['#_triplets'] = dataset_processed['graph'].map(
        lambda x: len(x.split("|"))
    )
    dataset_processed['#_text_tokens'] = dataset_processed['text'].map(
        lambda x: len(nltk.word_tokenize(x))
    )
    dataset_processed['#_graph_tokens'] = dataset_processed['graph'].map(
        lambda x: len(nltk.word_tokenize(x))
    )
    
    print("Data processed")
    
    return dataset_processed


def bio_event_filter_dataset(dataset):
    
    dataset_sorted = dataset.sort_values(
        by=['text', '#_triplets', '#_graph_tokens'], 
        ascending=False
    )
    dataset_filtered = dataset_sorted[
        dataset_sorted.duplicated(subset=['text'], keep='first') == False
    ]
    
    print("Data filtered")
    
    return dataset_filtered


def bio_event_save_dataset(dataset, frac_train=0.8, seed=41):
    train_set = dataset.sample(frac=frac_train, random_state=seed)
    val_set = dataset.drop(train_set.index).sample(frac=0.5, random_state=seed)
    test_set = dataset.drop(train_set.index).drop(val_set.index)
    
    train_set.to_csv('data/Bio_Event/processed/bio_event_e2t_train.csv', index=False)
    val_set.to_csv('data/Bio_Event/processed/bio_event_e2t_dev.csv', index=False)
    test_set.to_csv('data/Bio_Event/processed/bio_event_e2t_test.csv', index=False)
    
    print("Train size: {}; Val size: {}; Test size: {}".format(
        len(train_set),
        len(val_set),    
        len(test_set),
    ))
    
    print("Data saved")
    
    
