import torch
from torch import nn
from transformers import BertModel, AutoModel

class BertLabeler(nn.Module):
    def __init__(
            self,
            config,
            num_tags,
            pretrain_path=None
    ):
        super(BertLabeler, self).__init__()

        self.config = config
        self.num_tags = num_tags
        if pretrain_path == None:
            raise NotImplementedError
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.dropout = nn.Dropout(config.classifier_dropout)
        hidden_size = self.bert.pooler.dense.in_features

        #classes: yes, no for each tag
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 2, bias=True) for _ in range(int(num_tags))])

    def forward(
            self,
            input_ids,
            attention_mask,
    ):

        final_hidden = self.bert(
            input_ids, attention_mask=attention_mask
        )[0]
        # final_hidden = (batch_size, max_len, hidden_size)

        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)
        # cls_hidden = (batch_size, hidden_size)

        output = []
        for i in range(int(self.num_tags)):
            output.append(self.linear_heads[i](cls_hidden))

        return output
