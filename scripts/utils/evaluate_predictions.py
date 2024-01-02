import evaluate

def evaluate_predictions(test_set):
    
    print("Evaluating predictions...")
    
    rouge = evaluate.load('rouge')
    
    evaluations = []
    
    predictions = test_set["icl_predictions"]
    predictions_trunc = test_set['icl_predictions_trunc']
    references = test_set["triplets"]
    
    rouge_scores = rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=False
    )
    
    rouge_scores_trunc = rouge.compute(
            predictions=predictions_trunc,
            references=references,
            use_aggregator=False
    )
    
    test_set = test_set.assign(
        rouge1 = rouge_scores['rouge1'], 
        rouge2 = results['rouge2'],
        rougeL = results['rougeL'],
        rougeLsum = results['rougeLsum'],
        rouge1_trunc = results_trunc['rouge1'], 
        rouge2_trunc = results_trunc['rouge2'],
        rougeL_trunc = results_trunc['rougeL'],
        rougeLsum_trunc = results_trunc['rougeLsum'] 
    )
    
    rouge_scores = rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=True
    )
    
    rouge_scores_trunc = rouge.compute(
            predictions=predictions_trunc,
            references=references,
            use_aggregator=True
    )
        
    print("Predictions evaluated")
    
    print("Rouge scores: ", rouge_scores, "\nRouge scores truncated: ", rouge_scores_trunc)
        
    return rouge_scores, rouge_scores_trunc, test_set