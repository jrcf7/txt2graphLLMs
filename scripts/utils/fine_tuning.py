import nltk
import evaluate
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
from datasets import Dataset



class trainer_class:
    
    def __init__(
        self, 
        model_id,
        device
    ):
        self.model_id = model_id
        self.device = device
        self.metric = evaluate.load("rouge")
        
        if "t5" in model_id:
            self.tokenizer = T5Tokenizer.from_pretrained(model_id)
            self.model = T5ForConditionalGeneration.from_pretrained(model_id)
        elif "bart" in model_id or "knowgl" in model_id:
            self.tokenizer = BartTokenizer.from_pretrained(model_id)
            self.model = BartForConditionalGeneration.from_pretrained(model_id)
    
        self.model = self.model.to(device)


    def batch_tokenize(self, batch, tokenizer, max_text_length, max_graph_length):
    
        text, graph   = batch['text'], batch['graph']
        text_tokenized = tokenizer(
            text, padding="max_length", truncation=True, max_length=max_text_length
        )
        graph_tokenized = tokenizer(
            graph, padding="max_length", truncation=True, max_length=max_graph_length
        )

        batch = {k: v for k, v in text_tokenized.items()}
    
        # Ignore padding in the loss
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in l]
            for l in graph_tokenized["input_ids"]
        ]
    
        return batch


    def compute_metrics(self, eval_preds):
        
        tokenizer = self.tokenizer
        metric = self.metric
    
        preds, labels = eval_preds

        # Replace -100 as we can not predict them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds  = np.where(preds != -100, preds, tokenizer.pad_token_id)

        # decode preds and labels
        decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds  = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            use_stemmer=True
        )
    
        return result


    def initiate_trainer(
        self,
        train_set, 
        val_set, 
        training_args
    ):
        
        model=self.model
        tokenizer=self.tokenizer
    
        encoder_max_length = max(train_set["#_text_tokens"])
        decoder_max_length = max(val_set["#_graph_tokens"])
    
        train_batch_tokenized = Dataset.from_pandas(train_set).map(
            lambda batch: self.batch_tokenize(
                batch=batch, 
                tokenizer=tokenizer, 
                max_text_length=encoder_max_length, 
                max_graph_length=decoder_max_length
            ),
            batched=True,
            remove_columns=Dataset.from_pandas(train_set).column_names,
        )

        val_batch_tokenized = Dataset.from_pandas(val_set).map(
            lambda batch: self.batch_tokenize(
                batch=batch, 
                tokenizer=tokenizer, 
                max_text_length=encoder_max_length, 
                max_graph_length=decoder_max_length
            ),
            batched=True,
            remove_columns=Dataset.from_pandas(val_set).column_names,
        )
    
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, 
            model=model, 
            label_pad_token_id=tokenizer.pad_token_id
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_batch_tokenized,
            eval_dataset=val_batch_tokenized,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
        )
    
        return trainer