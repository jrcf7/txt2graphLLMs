from tqdm import tqdm

def build_single_prompt(train_set, N, seed):
        
    # Select the labelled examples for iCL
    icl_examples = train_set.sample(N, random_state=seed)
            
    prompt = "Convert the text into a sequence of triplets:\n"
    for text, graph in icl_examples[['text', 'graph']].values:
        prompt += "Text: {}\nGraph: {}\n".format(text, graph)
            
    return prompt
        
        
def build_prompts(train_set, test_set, N, seed):
        
    print("Building prompts...")
        
    prompt_list = []
        
    for text in tqdm(test_set['text'].values):
        seed += 1
        prompt = build_single_prompt(
                train_set=train_set, 
                N=N,
                seed=seed
        ) 
        prompt += f"Text: {text}\nGraph:"
        prompt_list.append(prompt)
        
    print("Prompts built")
        
    return prompt_list
        