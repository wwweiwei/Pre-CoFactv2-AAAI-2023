from transformers import AutoTokenizer, AutoModel, Swinv2Model
import pandas as pd
import logging
import ast
import argparse
import pickle
import sys
import os
from sklearn.metrics import f1_score
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

from model import FakeNet


transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


class MultiModalDataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()

        with open('../data/processed_{}.pickle'.format(mode), 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        claim_text, claim_image, document_text, document_image, claim_ocr, document_ocr, add_feature = self.data[idx]
        return (claim_text, claim_image, document_text, document_image, claim_ocr, document_ocr, add_feature)
        # claim_text, claim_image, document_text, document_image, category, claim_ocr, document_ocr, add_feature = self.data[idx]
        # return (claim_text, claim_image, document_text, document_image, torch.tensor(category), claim_ocr, document_ocr, add_feature)


if __name__ == '__main__':
    model_path = sys.argv[1]
    config = ast.literal_eval(open(model_path + '{}config'.format(sys.argv[2])).readline())
    set_seed(config['seed_value'])

    # df_val = pd.read_csv('../data/val.csv', index_col=0, sep='\t')[['claim', 'claim_image', 'document', 'document_image', 'Category']]
    # df_val = pd.read_csv('../data/test.csv', index_col=0, sep='\t')[['claim', 'claim_image', 'document', 'document_image']]
    # df_val['index'] = df_val.index

    category = {
        'Support_Multimodal': 0,
        'Support_Text': 1,
        'Insufficient_Multimodal': 2,
        'Insufficient_Text': 3,
        'Refute': 4
    }
    
    inverse_category = {
        0: 'Support_Multimodal',
        1: 'Support_Text',
        2: 'Insufficient_Multimodal',
        3: 'Insufficient_Text',
        4: 'Refute'
    }

    # df_val['Label'] = df_val['Category'].map(category)

    # load pretrained NLP model
    deberta_tokenizer = AutoTokenizer.from_pretrained(config['pretrained_text'])

    deberta = AutoModel.from_pretrained(config['pretrained_text'])
    if config['freeze_text']:
        for name, param in deberta.named_parameters():
            param.requires_grad = False
            # if 'adapter' not in name:
            #     param.requires_grad = False

    # vit_model = ViTModel.from_pretrained(config['pretrained_image'])
    vit_model = Swinv2Model.from_pretrained(config['pretrained_image'])
    if config['freeze_image']:
        for name, param in vit_model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

    fake_net = FakeNet(config)

    fake_net.load_state_dict(torch.load(model_path + '{}model'.format(sys.argv[2]), map_location=torch.device("cpu"))) #cuda:0")))
    vit_model.load_state_dict(torch.load(model_path + '{}vitmodel'.format(sys.argv[2]), map_location=torch.device("cpu"))) #cuda:0")))

    MAX_SEQUENCE_LENGTH = 512
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    deberta.to(device)
    vit_model.to(device)
    fake_net.to(device)

    # val_dataset = MultiModalDataset(mode='val')
    val_dataset = MultiModalDataset(mode='test')
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    # testing
    y_pred, y_true = [], []
    total_loss = 0
    with torch.no_grad():
        fake_net.eval(), deberta.eval(), vit_model.eval()
        for loader_idx, item in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            # claim_text, claim_image, document_text, document_image, label, claim_ocr, document_ocr, add_feature = item[0], item[1].to(device), item[2], item[3].to(device), item[4].to(device), list(item[5]), list(item[6]), item[7].to(device)
            claim_text, claim_image, document_text, document_image, claim_ocr, document_ocr, add_feature = item[0], item[1].to(device), item[2], item[3].to(device), list(item[4]), list(item[5]), item[6].to(device)

            # transform sentences to embeddings via DeBERTa
            input_claim = deberta_tokenizer(claim_text, truncation=True, padding=True, return_tensors="pt").to(device)
            output_claim_text = deberta(**input_claim).last_hidden_state

            input_document = deberta_tokenizer(document_text, truncation=True, padding=True, return_tensors="pt").to(device)
            output_document_text = deberta(**input_document).last_hidden_state

            output_claim_image = vit_model(claim_image).last_hidden_state
            output_document_image = vit_model(document_image).last_hidden_state

            predicted_output, concat_embeddings = fake_net(output_claim_text, output_claim_image, output_document_text, output_document_image, add_feature)
            # softmax = nn.Softmax(dim=1)
            # predicted_output = softmax(predicted_output)
            _, predicted_output = torch.topk(predicted_output, 1)

            if len(y_pred) == 0:
                y_pred = predicted_output.cpu().detach().tolist()
                # y_true = label.tolist()
            else:
                y_pred += predicted_output.cpu().detach().tolist()
                # y_true += label.tolist()

            torch.cuda.empty_cache()

    # f1 = round(f1_score(y_true, y_pred, average='weighted'), 5)
    
    # with open('record.csv', 'a') as config_file:
    #     config_file.write(model_path + ',' + str(f1))
    #     config_file.write('\n')

    answer = pd.DataFrame(y_pred, columns=['Category'])
    answer['Category'] = answer['Category'].map(inverse_category)
    answer.to_csv('./answer_test.csv')

    # answer = pd.DataFrame(y_pred, columns=category.keys())
    # answer['Category'] = answer['Category'].map(inverse_category)
    # answer.to_csv('{}answer.csv'.format(model_path))