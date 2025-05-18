import re
import pandas as pd
from torch.utils.data import Dataset


def text_prompt(prompt_template, arr_text):
    # Compose prompts
    arr_input_prompt = [prompt_template.format(eval_text=text) for text in arr_text]
    return arr_input_prompt


# Model training/inference
class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def add_empty_ann(arr, tweet_id, text):
    arr.append({
        'tweet_id': tweet_id,
        'text': text,
        'start': '-',
        'end': '-',
        'span': '-'
    })


def text_preds(arr_list_preds):
    return ['[' + ','.join(list_preds) + ']' for list_preds in arr_list_preds]


def extract_preds(arr_preds):
    arr_list_preds = []
    for model_pred in arr_preds:
        text_pred = model_pred.split('[')[-1].split(']')[0].strip()
        if text_pred:
            res = sorted(set(span.strip().lower() for span in text_pred.split(',') if span.strip()))
            arr_list_preds.append(res)
        else:
            arr_list_preds.append([])
    
    return arr_list_preds


def llm_format_preds(arr_preds, df_text_eval):
    assert len(arr_preds) == df_text_eval.shape[0]
    arr_pred_ann = []
    arr_bad_format = []
    for i, _ in enumerate(arr_preds):
        row = df_text_eval.iloc[i]
        tweet_id, text = row['tweet_id'], row['text']
        if arr_preds[i][0] != '[' or arr_preds[i][-1] != ']':
            arr_bad_format.append(i)
        text_pred = arr_preds[i].split('[')[-1].split(']')[0].strip()
        if len(text_pred) > 0:
            for span in text_pred.split(','):
                span = span.strip()  # just in case the model separated predictions using spaces before/after ','
                arr_match = list(re.finditer(re.escape(span.lower()), text.lower()))
                for m in arr_match:
                    start = m.start()
                    end = m.end()
                    assert text.lower()[start:end] == span.lower()
                    arr_pred_ann.append({
                        'tweet_id': tweet_id,
                        'text': text,
                        'start': start,
                        'end': end,
                        'span': text[start:end]                        
                    })
            # Add empty ann if none of the predicted spans matched
            if len(arr_pred_ann) == 0 or arr_pred_ann[-1]['tweet_id'] != tweet_id:
                add_empty_ann(
                    arr=arr_pred_ann,
                    tweet_id=tweet_id,
                    text=text
                )

        else:
            add_empty_ann(
                arr=arr_pred_ann,
                tweet_id=tweet_id,
                text=text
            )

    df_preds = pd.DataFrame(arr_pred_ann).drop_duplicates(
        ['tweet_id', 'start', 'end'], keep='first'
    )

    return df_preds, arr_bad_format


def filter_token(span, arr_filter):
    """Check if any token from the span is in the filter list."""
    return any(token in arr_filter for token in span.lower().split())


def lexicon_filter_preds(arr_model_preds, arr_filter, filter_func=filter_token):
    """Filter model predictions based on a specified array of drugs and filter function."""
    arr_filter_preds = []
    for text in arr_model_preds:
        # Extract drug spans from model predictions
        model_text_pred = text.split('[')[-1].split(']')[0].strip()
        if model_text_pred:
            # Apply the filter function to each span in the model_text_pred
            arr_preds = [
                span for span in model_text_pred.split(',')
                if filter_func(span, arr_filter)
            ]
        else:
            arr_preds = []
        
        arr_filter_preds.append('[' + ','.join(arr_preds) + ']')
    return arr_filter_preds


class PredsFormatter():
    def __init__(
        self, df_text, tokenizer, dataset, arr_lexicon=None
    ):
        self.df_text = df_text
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.arr_lexicon = arr_lexicon

    def format_preds(self, pred, lexicon_filter=True):
        arr_text_preds = self.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        if lexicon_filter and self.arr_lexicon:
            arr_text_preds = lexicon_filter_preds(
                arr_model_preds=arr_text_preds,
                arr_filter=self.arr_lexicon,
                filter_func=filter_token
            )
        df_preds, arr_bad_format = llm_format_preds(
            arr_preds=arr_text_preds, df_text_eval=self.df_text
        )

        return df_preds, arr_bad_format, arr_text_preds

