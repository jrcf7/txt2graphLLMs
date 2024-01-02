def post_process_predictions(predictions):
    
    predictions.map(lambda x: x.split(')]')[0] + ')]' if ')]' in x else x)
    
    predictions.map(lambda x: x.lstrip())

    return predictions