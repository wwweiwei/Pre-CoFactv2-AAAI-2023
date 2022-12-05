import pandas as pd
import logging
import argparse
import pickle
import os
import gc
import yaml
from tqdm import tqdm
from transformers import ViTModel, Swinv2Model
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_scheduler
from sklearn.metrics import f1_score
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from pytorch_metric_learning import losses

from model import FakeNet


transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--output_folder_name",
                        type=str,
                        help="path to save model")
    opt.add_argument("--config",
                        type=str,
                        help="config path")
    config = vars(opt.parse_args())
    return config


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
        # + 1 for 2022 data (not sure why 2023 not need)
        claim_text, claim_image, document_text, document_image, category, claim_ocr, document_ocr, add_feature = self.data[idx]
        return (claim_text, claim_image, document_text, document_image, torch.tensor(category), claim_ocr, document_ocr, add_feature)


def save(model, vit_model, config, epoch=None):
    output_folder_name = config['output_folder_name']
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    if epoch is None:
        model_name = output_folder_name + 'model'
        vit_model_name = output_folder_name + 'vitmodel'
        config_name = output_folder_name + 'config'
    else:
        model_name = output_folder_name + str(epoch) + 'model'
        vit_model_name = output_folder_name + str(epoch) + 'vitmodel'
        config_name = output_folder_name + str(epoch) + 'config'
    
    torch.save(model.state_dict(), model_name)
    torch.save(vit_model.state_dict(), vit_model_name)
    with open(config_name, 'w') as config_file:
        config_file.write(str(config))


if __name__ == '__main__':
    input_argument = get_argument()
    with open(input_argument['config'], "r") as file:
        config = yaml.safe_load(file)

    config['output_folder_name'] = input_argument['output_folder_name']
    set_seed(config['seed_value'])

    # load pretrained NLP model
    deberta_tokenizer = AutoTokenizer.from_pretrained(config['pretrained_text'])
    deberta = AutoModel.from_pretrained(config['pretrained_text'])
    if config['freeze_text']:
        for name, param in deberta.named_parameters():
            param.requires_grad = False
            # if 'adapter' not in name:
            #     param.requires_grad = False

    vit_model = Swinv2Model.from_pretrained(config['pretrained_image'])
    if config['freeze_image']:
        for name, param in vit_model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

    fake_net = FakeNet(config)

    fake_net.load_state_dict(torch.load('/home/wei/ssd2/factify/yao/model/20221201-131212_/10model', map_location=torch.device(f"cuda:{config['device']}")))
    vit_model.load_state_dict(torch.load('/home/wei/ssd2/factify/yao/model/20221201-131212_/10vitmodel', map_location=torch.device(f"cuda:{config['device']}")))

    criterion = torch.nn.CrossEntropyLoss()
    fake_net_optimizer = AdamW(fake_net.parameters(), lr=config['lr'])

    device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # loss_func = losses.SupConLoss().to(device)

    deberta.to(device)
    vit_model.to(device)
    fake_net.to(device)
    criterion.to(device)

    train_dataset = MultiModalDataset(mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    val_dataset = MultiModalDataset(mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)

    scheduler = get_scheduler("linear", fake_net_optimizer, num_warmup_steps=int(config['epochs']*len(train_dataloader)*0.1), num_training_steps=config['epochs']*len(train_dataloader))

    print(f"{sum(p.numel() for p in deberta.parameters() if p.requires_grad)}")
    print(f"{sum(p.numel() for p in vit_model.parameters() if p.requires_grad)}")
    print(f"{sum(p.numel() for p in fake_net.parameters() if p.requires_grad)}")

    # training
    pbar = tqdm(range(config['epochs']), desc='Epoch: ')
    for epoch in pbar:
        fake_net.train()
        total_loss, best_val_f1, total_ce, total_scl = 0, 0, 0, 0
        for loader_idx, item in enumerate(train_dataloader): 
            fake_net_optimizer.zero_grad()
            claim_text, claim_image, document_text, document_image, label, claim_ocr, document_ocr, add_feature = list(item[0]), item[1].to(device), list(item[2]), item[3].to(device), item[4].to(device), list(item[5]), list(item[6]), item[7].to(device)

            # transform sentences to embeddings via DeBERTa
            input_claim = deberta_tokenizer(claim_text, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
            output_claim_text = deberta(**input_claim).last_hidden_state

            input_document = deberta_tokenizer(document_text, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
            output_document_text = deberta(**input_document).last_hidden_state

            input_claim_ocr = deberta_tokenizer(claim_ocr, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
            output_claim_ocr = deberta(**input_claim_ocr).last_hidden_state

            input_document_ocr = deberta_tokenizer(document_ocr, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
            output_document_ocr = deberta(**input_document_ocr).last_hidden_state

            output_claim_image = vit_model(claim_image).last_hidden_state
            output_document_image = vit_model(document_image).last_hidden_state

            predicted_output, concat_embeddings = fake_net(output_claim_text, output_claim_image, output_document_text, output_document_image, add_feature)
            
            ce_loss = criterion(predicted_output, label)
            # scl_loss = loss_func(concat_embeddings, label)
            # loss = config['loss_weight'] * ce_loss + (1 - config['loss_weight']) * scl_loss
            loss = ce_loss
            loss.backward()
            fake_net_optimizer.step()
            scheduler.step()

            current_loss = loss.item()
            total_loss += current_loss
            total_ce += ce_loss.item()
            # total_scl += scl_loss.item()

            pbar.set_description("Loss: {}".format(round(current_loss, 3)), refresh=True)

            # if loader_idx == 2:
            #     break

        # print(f'total loss: {round(total_loss/len(train_dataloader), 4)} | ce: {round(total_ce/len(train_dataloader), 4)} | scl: {round(total_scl/len(train_dataloader), 4)}')
        print(f'total loss: {round(total_loss/len(train_dataloader), 4)} | ce: {round(total_ce/len(train_dataloader), 4)}')

        del claim_text, claim_image, document_text, document_image, label,  claim_ocr, document_ocr, add_feature, input_claim, output_claim_text, input_document, output_document_text, output_claim_image, output_document_image, predicted_output, loss
        gc.collect()
        with torch.cuda.device(f"cuda:{config['device']}"):
            torch.cuda.empty_cache()
        save(fake_net, vit_model, config, epoch=epoch)

        if epoch % config['eval_per_epochs'] == 0:
            # testing
            with torch.no_grad():
                y_pred, y_true = [], []
                fake_net.eval(), deberta.eval(), vit_model.eval()
                for loader_idx, item in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                    claim_text, claim_image, document_text, document_image, label, claim_ocr, document_ocr, add_feature = list(item[0]), item[1].to(device), list(item[2]), item[3].to(device), item[4].to(device), list(item[5]), list(item[6]), item[7].to(device)

                    # transform sentences to embeddings via DeBERTa
                    input_claim = deberta_tokenizer(claim_text, truncation=True, padding=True, return_tensors="pt").to(device)
                    output_claim_text = deberta(**input_claim).last_hidden_state

                    input_document = deberta_tokenizer(document_text, truncation=True, padding=True, return_tensors="pt").to(device)
                    output_document_text = deberta(**input_document).last_hidden_state

                    output_claim_image = vit_model(claim_image).last_hidden_state
                    output_document_image = vit_model(document_image).last_hidden_state

                    predicted_output, concat_embeddings = fake_net(output_claim_text, output_claim_image, output_document_text, output_document_image, add_feature)
                    
                    _, predicted_label = torch.topk(predicted_output, 1)

                    if len(y_pred) == 0:
                        y_pred = predicted_label.cpu().detach().flatten().tolist()
                        y_true = label.tolist()
                    else:
                        y_pred += predicted_label.cpu().detach().flatten().tolist()
                        y_true += label.tolist()

                f1 = round(f1_score(y_true, y_pred, average='weighted'), 5)

                if f1 >= best_val_f1:
                    best_val_f1 = f1
                    save(fake_net, vit_model, config, epoch=epoch)

                with open(config['output_folder_name'] + 'record', 'a') as config_file:
                    config_file.write(str(epoch) + ',' + str(round(total_loss/len(train_dataloader), 5)) + ',' + str(f1))
                    config_file.write('\n')

    config['total_loss'] = total_loss
    config['val_f1'] = best_val_f1
    save(fake_net, vit_model, config)