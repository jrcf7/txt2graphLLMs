# txt2graphLLMs
On General and Biomedical Text-to-Graph Large Language Models.


## Environment setup
We recommend using conda as package manager. Please find the environment requirements in `environment.yml`. Note that this is an environment that evolved from an earlier project and as such is not a minimal working product. 

To create the environment, run the following commands. 
`
conda env create -f environment.yml
conda activate txt2graph_llms
`


## Load and pre-process data
Load the Web NLG (version 3.0) dataset and pre-process both Web NLG and Bio Event by running the following commands. The results can be found in the `\data` folder. 
`
python scripts/pre_process_data.py
`


## Run experiments for in-context learning (iCL) and Fine-Tuning (FT)
To run the main experiments (using 8 in-context examples), run the following commands. The results can be found in the `\output` folder. Note that you will be prompted for your Hugging Face login key, which requires read access for the iCL experiments and write access for the FT experiments (to upload fine-tuned models to Hugging Face). In the latter case you will also be prompted for your Hugging Face username. All fine-tuned models are saved both locally and in your personal Hugging Face repository. 
`
python scripts/run_experiments_icl.py --models all --icl_examples 8 --datasets web_nlg bio_event
python scripts/run_experiments_ft.py --models all --datasets web_nlg bio_event
`

To run the next set of experiments varying the amount of in-context examples, run the following commands.
`
python scripts/run_experiments_icl.py --models mistral --icl_examples 0 2 4 8 16 32 --datasets web_nlg bio_event
python scripts/run_experiments_icl.py --models llama --icl_examples 0 2 4 8 --datasets web_nlg bio_event
`

In both cases, if you desire to run a single model on a single dataset, simply modify the options in the following lines
`
python scripts/run_experiments_icl.py --models mistralai/Mistral-7B-v0.1 mistralai/Mistral-7B-Instruct-v0.1 Open-Orca/Mistral-7B-OpenOrca meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-13b-chat-hf epfl-llm/meditron-7b --icl_examples 0 2 4 8 16 32 --datasets web_nlg bio_event --batch_size 24 --seed 41 --debug False
python scripts/run_experiments_ft.py --models t5-small t5-base t5-large facebook/bart-large ibm/knowgl-large --datasets web_nlg bio_event --batch size 24 --epochs 10 --seed 41 --debug False
`


## Demo notebook 
Checkout `demo_results.ipynb` to recreate the graphs in the main body of our paper. Note that running the previous commands is necessary to obtain the needed model output first. Furthermore, this repository is a cleaned-up version of the codebase used to produce the graphs in the paper and although the results should be perfectly replicable, this has not been tested. 


## Code
The main scripts can be found in `\scripts` with various util functions in `\scripts\utils`.


## Citing our work

