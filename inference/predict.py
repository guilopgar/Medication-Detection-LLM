import argparse
import pandas as pd
import numpy as np
import pickle
from transformers import set_seed, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import utils
from datetime import timedelta
import time


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def predict_tweets(df_data, arr_lexicon, tokenizer, model):
    prompt_template = """You are given a tweet followed by a specific question asking about the content of the tweet.
Your objective is to identify and list any drug names, medications, or dietary supplements mentioned in the tweet.
If one or more are mentioned, list each distinctly, separated by a comma. If none are mentioned, return an empty list [].
Input: Tweet: {eval_text} Question: What are the drugs, medications or dietary supplements mentioned in the tweet?
Output: """

    # Input data
    encodings = tokenizer(
        utils.text_prompt(
            prompt_template=prompt_template,
            arr_text=df_data["text"].tolist()
        ),
        truncation=False,
        padding=True,
        return_tensors="pt"
    )
    dataset = utils.CustomDataset(
        encodings=encodings
    )
    preds_format = utils.PredsFormatter(
        df_text=df_data,
        tokenizer=tokenizer,
        dataset=dataset,
        arr_lexicon=arr_lexicon
    )

    # Model trainer
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors='pt'
    )
    training_args = Seq2SeqTrainingArguments(
        tf32=True,
        dataloader_num_workers=4,
        output_dir="./",
        predict_with_generate=True,
        disable_tqdm=True,
        per_device_eval_batch_size=32,
        seed=0
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator
    )

    # Model predictions
    start_time = time.time()
    preds = trainer.predict(preds_format.dataset)
    end_time = time.time()
    print("Total inference time:", str(timedelta(seconds=end_time - start_time)))

    # Format predictions
    df_preds, arr_bad_format, arr_preds = preds_format.format_preds(pred=preds)
    print("Number of badly formatted predictions:", len(arr_bad_format))
    if len(arr_bad_format) > 0:
        print("Badly formatted predictions:")
        print(np.array(arr_preds)[arr_bad_format])
    
    return df_preds


def main():
    parser = argparse.ArgumentParser(description="Medication Mention Detection using Flan-T5")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True, help="Hugging Face model name or path"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file containing tweets"
    )
    parser.add_argument(
        "--lexicon_path",
        type=str,
        default=None,
        help="Optional path to lexicon file for filtering"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the output predictions CSV"
    )

    args = parser.parse_args()

    # Load input data
    df_data = pd.read_csv(args.input_csv)
    lexicon = None
    if args.lexicon_path:
        with open(args.lexicon_path, 'rb') as file:
            lexicon = pickle.load(file)

    # GPU configuration
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available()

    # Load model and tokenizer
    set_seed(0)
    tokenizer, model = load_model_and_tokenizer(args.model_name)

    # Run predictions
    df_predictions = predict_tweets(
        df_data=df_data,
        arr_lexicon=lexicon,
        tokenizer=tokenizer,
        model=model
    )

    # Save predictions
    df_predictions.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
