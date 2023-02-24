import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, questions, multilabels_df=None):
        self.questions = questions
        self.multilabels_df = multilabels_df if multilabels_df is not None else None

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        question = str(self.questions[idx])
        multilabel = self.multilabels_df.iloc[idx, :].to_numpy() if self.multilabels_df is not None else None
        multilabel = torch.LongTensor(multilabel) if multilabel is not None else None

        return {
            'question': question,
            'multilabel': multilabel
        }

        
class TokenizerWrapper():

    def __init__(
        self,
        tokenizer,
        max_length,
    ):

        self.tokenizer = tokenizer
        self.max_length = max_length

    def collate(self, samples):

        questions = [sample['question'] for sample in samples]

        encoding = self.tokenizer(
            text=questions, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        if samples[0]['multilabel'] is not None:
            multilabels = [sample['multilabel'] for sample in samples]
            multilabels = torch.stack(multilabels, dim=0)
        else:
            multilabels = None
    
        return {
            'input_ids': torch.LongTensor(encoding['input_ids']),
            'attention_mask': torch.LongTensor(encoding['attention_mask']),
            'multilabels': torch.LongTensor(multilabels) if multilabels is not None else None,
        }