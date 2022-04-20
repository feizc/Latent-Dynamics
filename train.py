
from transformers import GPT2Tokenizer, data
from argparse import ArgumentParser
import logging
import random 
import torch 
import numpy as np 


from dataset import build_input_from_story 
logger = logging.getLogger(__file__)


def main(): 
    parser = ArgumentParser()  
    parser.add_argument("--data_path", type=str, default="./dataset/train.json", help="Path of the dataset") 
    parser.add_argument("--tokenizer_path", type=str, default="ckpt/gpt2", help="Path of the tokenizer") 
    args = parser.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0) 

    logging.basicConfig(level=logging.INFO)


    logger.info('Prepare tokenizer and model') 
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path) 

    logger.info('Prepare datasets') 
    data_list = build_input_from_story(args.data_path) 
    print(data_list)










if __name__ == "__main__":
    main()