from matplotlib import image
from transformers import GPT2Tokenizer, data
from argparse import ArgumentParser
import logging
import random 
import torch 
import numpy as np 


from dataset import build_input_from_story, VisualStoryDataset 
import clip 
logger = logging.getLogger(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"  

SPECIAL_TOKENS = ["[bos]", "[eos]",] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]"}


def main(): 
    parser = ArgumentParser()  
    parser.add_argument("--data_path", type=str, default="./dataset/train.json", help="Path of the dataset") 
    parser.add_argument("--tokenizer_path", type=str, default="ckpt/gpt2", help="Path of the tokenizer") 
    parser.add_argument("--clip_path", type=str, default="ckpt/clip/ViT-B-32.pt", help="Path of the clip model") 
    parser.add_argument('--batch_size', type=int, default=1)
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
    print(data_list)
    dataset = VisualStoryDataset(data_list, tokenizer, image_encoder, preprocess, device) 
    image_features_list, history_txt_list, caption_ids = dataset[2] 
    print(len(image_features_list))











if __name__ == "__main__":
    main()