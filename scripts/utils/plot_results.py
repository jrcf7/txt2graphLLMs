import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import re

def gather_results(N=[0, 2, 4, 8, 16, 32], batch_size=24, seed=41, debug=False):

    models = {
        "t5-small": ["T5", 60.5, "Fine-tuned"],
        "t5-base": ["T5", 223, "Fine-tuned"],
        "t5-large": ["T5", 738, "Fine-tuned"],
        "t5-3b": ["T5", 3000, "Fine-tuned"],
        "facebook_bart-large": ["BART", 406, "Fine-tuned"],
        "ibm_knowgl-large": ["BART + REBEL", 406, "Fine-tuned"],
        "mistralai_Mistral-7B-v0.1": ["Mistral-v1", 7000, "iCL"],
        "mistralai_Mistral-7B-Instruct-v0.1": ["Mistral-v1 + Instruction", 7000, "iCL"],
        "Open-Orca_Mistral-7B-OpenOrca": ["Mistral-v1 + OpenOrca", 7000, "iCL"],
        "meta-llama_Llama-2-7b-chat-hf": ["Llama-2 + Instruction", 7000, "iCL"],
        "meta-llama_Llama-2-13b-chat-hf": ["Llama-2 + Instruction", 13000, "iCL"],
        "epfl-llm_meditron-7b": ["Llama-2 + Meditron", 7000, "iCL"],
    }

    datasets = {
        "web_nlg": "Web NLG",
        'bio_event': 'Bio Event'
    }

    results = pd.DataFrame(columns=['model_id', 'Model', 'Size', 'Method', 'N', 'Dataset',
                                'rouge_1', 'rouge_2', 'rouge_l', 
                                'rouge_1_postprocessed', 'rouge_2_postprocessed', 'rouge_l_postprocessed'])
    
    idx = 0
    for c, (model_id, (model, model_size, method)) in enumerate(models.items()):
        for dataset_id, dataset in datasets.items():
            for n in N:
                try:
                    test_set = pd.read_csv(
                        '../output/test_set_{}_{}_N-{}_batch_size-{}_seed-{}_debug-{}.csv'.format(model_id, dataset_id, n, batch_size, seed, debug),
                        index_col=0)
                    if 'model_output_postprocessed' in test_set.columns:
                        rouge = test_set[['rouge_1', 'rouge_2', 'rouge_l',  
                                'rouge_1_postprocessed', 'rouge_2_postprocessed', 'rouge_l_postprocessed']].mean()
                        results.loc[idx] = [model_id] + [model] + [model_size] + [method] + [n] + [dataset] + list(rouge.values)
                    else:
                        rouge = test_set[['rouge_1', 'rouge_2', 'rouge_l']].mean()
                        results.loc[idx] = [model_id] + [model] + [model_size] + [method] + [None] + [dataset] + list(rouge.values) + [None] + [None] + [None]
                        idx += 1
                except Exception as E:
                    continue

    results.to_csv('../output/main_results_batch_size-{}_seed-{}_debug-{}.csv'.format(batch_size, seed, debug), index=False)
    
    results_pivot_rouge = results[list(results.columns[6:])].melt(value_name='Score')
    results_pivot_rouge = results_pivot_rouge.rename(columns={"variable": "metric_id"})
    results_pivot_rouge = results_pivot_rouge.reset_index(drop=True)

    results_pivot_specs = pd.concat(
        [results[list(results.columns[:6])]] * int((len(results_pivot_rouge)/len(results))), 
        ignore_index=True
    )

    results_pivot = pd.concat([results_pivot_rouge, results_pivot_specs], axis=1)
    results_pivot = results_pivot.reset_index(drop=True)

    results_pivot["Model Output"] = ["Heuristic" if 'postprocessed' in m else 'Standard' for m in results_pivot["metric_id"]]
    results_pivot["Metric"] = [m.split("_")[0].capitalize() + "-" + m.split("_")[1].capitalize() for m in results_pivot["metric_id"]]
    results_pivot.loc[results_pivot['Model Output']=='Heuristic', 'Method'] = results_pivot.loc[results_pivot['Model Output']=='Heuristic', 'Method'] + ' + Heuristic'
    results_pivot.drop(results_pivot[results_pivot['Score']==0].index, inplace = True)

    results_pivot.to_csv('../output/main_results_pivot_batch_size-{}_seed-{}_debug-{}.csv'.format(batch_size, seed, debug), index=False)
    
    return results, results_pivot


def scatterplot_models(results_pivot, N=[8]):

    sns.set_context("talk")
    plt.figure(figsize=(15,10))

    g = sns.relplot(
        data=results_pivot[
            (results_pivot["Metric"] == "Rouge-L") &
            (results_pivot['N'].isin(N + [None]))
        ],
        y="Score",
        x="Size",
        col="Dataset",
        style="Model",
        hue="Method",
        palette="Set1",
        kind="scatter",
        s=200,
        alpha=.7,
    )

    g.set(xlabel="Model Size (M Parameters)", ylabel="Rouge-L Score")
    g.set_titles(template='{col_name}')
    plt.xscale('log')
    g.fig.axes[1].set_ylabel(None)
    

def lineplot_metrics(results_pivot, N=[8]):

    sns.set_context("talk")
    plt.figure(figsize=(15,10))

    g = sns.relplot(
        data=results_pivot[(results_pivot["Method"] != "iCL") &  
                       (results_pivot['N'].isin(N + [None]))],
        y="Score",
        x="Size",
        col="Dataset",
        hue="Metric",
        style='Metric',
        errorbar=("se", 1),
        linewidth = 2,
        err_style='bars',
        err_kws={'capsize':7,
             "linewidth":2},
        markers=True,
        kind="line",
        palette="Set1",
    )
    
    plt.xscale('log')
    g.set(xlabel="Model Size (M Parameters)")
    g.set_titles(template='{col_name}')
    

def lineplot_incontext_examples(results_pivot, model_id='Open-Orca_Mistral-7B-OpenOrca', N=[0, 2, 4, 8, 16, 32]):

    sns.set_context("talk")
    plt.figure(figsize=(15,10))

    g = sns.relplot(
        data=results_pivot[
            results_pivot["model_id"].isin([model_id]) & 
            results_pivot["Metric"].isin(["Rouge-L"]) &
            results_pivot["N"].isin(N)
        ],
        x="N",
        y="Score",
        hue="Model Output",
        style="Model Output",
        markers=True,
        linewidth = 2,
        kind="line",
        col="Dataset",
        facet_kws={'sharey': True, 'sharex': True},
        palette="Set1",
    )

    # Rename ax and suptitles
    g.set_titles(template='{col_name}')
    plt.xticks([0, 2, 4, 8, 16, 32])
    # flatten axes into a 1-d array
    axes = g.axes.flatten()

    best_encod_dec_scores = [0.679060, 0.636729]
    # iterate through the axes
    for i, ax in enumerate(axes):
        ax.axhline(best_encod_dec_scores[i], ls='--', c='gray')
    
    g.fig.axes[0].set_ylabel("Rouge-L Score")
    


def pct_graph_tokens(seq):
    seq_text = re.split('\[\(.*\)\]', seq)
    seq_graph = seq
    for i in seq_text:
        seq_graph = seq_graph.replace(i, '')
    seq_text = ''.join(seq_text)
    
    pct_graph_tokens = len(word_tokenize(seq_graph))/(len(word_tokenize(seq_text))+len(word_tokenize(seq_graph)))
    
    return pct_graph_token

def nr_tokens(seq):
    return len(word_tokenize(seq))
    
def boxplot_fct(N, dataset_id, model_id, fct, post_processed=True, batch_size=24, seed=41, debug=False):
    pred = pd.DataFrame(columns=N)
    for n in N:
        try:
            test_set = pd.read_csv('../output/test_set_{}_{}_N-{}_batch_size-{}_seed-{}_debug-{}.csv'.format(model_id, dataset_id, n, batch_size, seed, debug))
            pred[n] = test_set['model_output' if post_processed else 'model_output_postprocessed'].apply(fct)
        except Exception as E:
            continue
    pred["target"] = test_set["graph"].apply(fct)
    g = sns.boxplot(data=pred)
    g.set(
        xlabel='N', 
        ylabel="% Graph Output" if fct==pct_graph_tokens else "# Tokens", 
        title="Web NLG" if dataset_id=='web_nlg' else "Bio Event"
    )
    
    plt.show()