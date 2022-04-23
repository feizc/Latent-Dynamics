from transformers import GPT2Tokenizer, AdamW
from argparse import ArgumentParser
import logging
import random 
import torch 
import numpy as np 


from dataset import build_input_from_story, VisualStoryDataset 
import clip 
from model import MultiCaptionGeneratorConfig, MutiCaptionGenerator


logger = logging.getLogger(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"  

SPECIAL_TOKENS = ["[bos]", "[eos]",] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]"}


def build_input(image_features_list, history_txt_list, caption_ids, model, args): 
    input_list = [] 
    for i in range(len(history_txt_list)): 
        t_image_features = image_features_list[i].to(device)
        t_image_embs = model.visual_project(t_image_features).view(args.batch_size, args.prefix_length, -1)
        input_list.append(t_image_embs)
        t_txt_ids = history_txt_list[i] 
        t_txt_embs = model.captioner.wte(t_txt_ids) 
        input_list.append(t_txt_embs) 
    t_image_features = image_features_list[-1].to(device)
    t_image_embs = model.visual_project(t_image_features).view(args.batch_size, args.prefix_length, -1)
    input_list.append(t_image_embs) 

    caption_length = caption_ids.size(1)
    t_txt_embs = model.captioner.wte(caption_ids) 
    input_list.append(t_txt_embs) 
    input_embs = torch.cat(input_list, dim=1) 
    total_length = input_embs.size(1)
    labels = torch.zeros((1, total_length)) - 100
    labels[:,-caption_length+1:] = caption_ids[:, 1:]
    return input_embs, labels 



def train(model, optimizer, dataset, args): 
    model.train() 
    for idx, (image_features_list, history_txt_list, caption_ids) in enumerate(dataset): 
        input_embs, labels = build_input(image_features_list, history_txt_list, caption_ids, model, args)
        logits = model(input_embs=input_embs) 
        print(logits.size())



def main(): 
    parser = ArgumentParser()  
    parser.add_argument("--data_path", type=str, default="./dataset/train.json", help="Path of the dataset") 
    parser.add_argument("--tokenizer_path", type=str, default="ckpt/gpt2", help="Path of the tokenizer") 
    parser.add_argument("--clip_path", type=str, default="ckpt/clip/ViT-B-32.pt", help="Path of the clip model") 
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--prefix_size', type=int, default=512)
    parser.add_argument('--prefix_length', type=int, default=10) 
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0) 

    logging.basicConfig(level=logging.INFO)


    logger.info('Prepare tokenizer and model') 
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)  
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 

    image_encoder, preprocess = clip.load(args.clip_path, device=device) 


    logger.info('Prepare datasets') 
    data_list = build_input_from_story(args.data_path) 
    # logger.info(data_list)
    dataset = VisualStoryDataset(data_list, tokenizer, image_encoder, preprocess, device) 

    model_config = MultiCaptionGeneratorConfig() 
    model = MutiCaptionGenerator(model_config, args, tokenizer)  
    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=args.lr)

    logger.info('Model training') 
    train(model, optimizer, dataset, args)













if __name__ == "__main__":
    main()