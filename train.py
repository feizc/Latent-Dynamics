from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from argparse import ArgumentParser 
from torch.nn import functional as F
import logging
import random 
import torch 
import numpy as np 
from tqdm import tqdm 
import os 

from dataset import build_input_from_story, VisualStoryDataset, tokenize
import clip 
from model import MultiCaptionGeneratorConfig, MutiCaptionGenerator
from utils import accuracy_compute 


logger = logging.getLogger(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"  


SPECIAL_TOKENS = ["[bos]", "[eos]", "[dyn]", "[img]", "[txt]"] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]", 'additional_special_tokens': ["[dyn]", "[img]", "[txt]"],}


def build_input(image_features_list, history_txt_list, caption_ids, model, args): 
    bos, eos, dyn, img, txt = [x[0] for x in tokenize(SPECIAL_TOKENS, model.tokenizer)]
    input_list = [] 
    sentence_index = [-1] 
    token_type_list = []

    cur_index = 0 
    input_combine_list = history_txt_list + [caption_ids]
    for i in range(len(input_combine_list)): 
        t_image_features = image_features_list[i].to(device)
        t_image_embs = model.visual_project(t_image_features).view(args.batch_size, args.prefix_length, -1) # (bsz, pre_len, model_d)
        input_list.append(t_image_embs) 
        token_type_list.extend([img]*t_image_embs.size(1))
        cur_index += t_image_embs.size(1)
        
        t_txt_ids = input_combine_list[i] 
        t_txt_embs = model.captioner.wte(t_txt_ids) 
        input_list.append(t_txt_embs) # (bsz, seq_len, model_d) 
        token_type_list.extend([img]*t_txt_embs.size(1))
        cur_index += t_txt_embs.size(1)
        sentence_index.append(cur_index - 1)

    input_embs = torch.cat(input_list, dim=1) 
    
    
    total_length = input_embs.size(1)
    labels = torch.zeros((1, total_length)) - 100 
    for i in range(len(sentence_index)-1): 
        start = sentence_index[i] + args.prefix_length + 1
        end = sentence_index[i+1]
        labels[:, start:end] = input_combine_list[i][:, :-1]

    labels = labels.long()
    
    sentence_index = torch.LongTensor(sentence_index) 
    token_types = torch.LongTensor(token_type_list)
    return input_embs, labels, sentence_index, token_types



def train(model, optimizer, scheduler, dataset, args): 
    model.train() 
    running_loss = .0 
    progress = tqdm(total=len(dataset), desc='visual storytelling training') 
    for idx, (image_features_list, history_txt_list, caption_ids) in enumerate(dataset): 
        model.zero_grad()
        input_embs, labels, sentence_index, token_types = build_input(image_features_list, history_txt_list, caption_ids, model, args)
        loss = model(input_embs=input_embs, token_type_ids=token_types, sentence_index=sentence_index, labels=labels) 
        loss.backward() 
        optimizer.step() 
        scheduler.step() 
        optimizer.zero_grad() 
        running_loss += loss.item() 
        progress.set_postfix({"loss": running_loss / (idx + 1)})
        progress.update() 
        if idx % 10000 == 0:
                torch.save({
                    'torch_rng_state': torch.get_rng_state(),
                    # 'cuda_rng_state': torch.cuda.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    }, os.path.join(args.output_dir, "latest.pt"),
                )
        break
    progress.close() 
    return running_loss


def eval(model, dataset, args): 
    model.eval() 
    running_loss = .0 
    with torch.no_grad(): 
        progress = tqdm(total=len(dataset), desc='visual storytelling evaluation') 
        for idx, (image_features_list, history_txt_list, caption_ids) in enumerate(dataset): 
            input_embs, labels, sentence_index, token_types = build_input(image_features_list, history_txt_list, caption_ids, model, args)
            loss = model(input_embs=input_embs, token_type_ids=token_types, sentence_index=sentence_index, labels=labels) 
            running_loss += loss.item() 
            progress.set_postfix({"loss": running_loss / (idx + 1)})
            progress.update() 
            break 
        progress.close() 
    return running_loss / len(dataset)




def main(): 
    parser = ArgumentParser()  
    parser.add_argument("--train_data_path", type=str, default="./dataset/train.json", help="Path of the training dataset") 
    parser.add_argument("--val_data_path", type=str, default="./dataset/val.json", help="Path of the validation dataset") 
    parser.add_argument("--tokenizer_path", type=str, default="ckpt/gpt2", help="Path of the tokenizer") 
    parser.add_argument("--clip_path", type=str, default="ckpt/clip/ViT-B-32.pt", help="Path of the clip model")
    parser.add_argument("--output_dir", type=str, default="ckpt/visual_storytelling", help="Path of the saving model")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--prefix_size', type=int, default=512)
    parser.add_argument('--prefix_length', type=int, default=10) 
    parser.add_argument('--warmup_steps', type=int, default=5000) 
    parser.add_argument('--epochs', type=int, default=30) 
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate") 
    parser.add_argument('--log_path', type=str, default='log/output.log', help='Log file path')
    args = parser.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0) 

    logging.basicConfig(level=logging.INFO, filename=args.log_path)


    logger.info('Prepare tokenizer and model') 
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)  
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 

    image_encoder, preprocess = clip.load(args.clip_path, device=device) 


    logger.info('Prepare datasets') 
    train_data_list = build_input_from_story(args.train_data_path) 
    val_data_list = build_input_from_story(args.val_data_path) 
    # logger.info(data_list)
    train_dataset = VisualStoryDataset(train_data_list, tokenizer, image_encoder, preprocess, device) 
    val_dataset = VisualStoryDataset(val_data_list, tokenizer, image_encoder, preprocess, device) 

    model_config = MultiCaptionGeneratorConfig() 
    model = MutiCaptionGenerator(model_config, args, tokenizer)  
    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_dataset)
    )
    
    logger.info('Model training') 
    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, scheduler, train_dataset, args) 
        val_loss = eval(model, val_dataset, args)



if __name__ == "__main__":
    main()