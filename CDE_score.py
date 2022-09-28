import torch 
import torch.nn as nn
from argparse import ArgumentParser 
import logging 
from transformers import GPT2Tokenizer
from utils import SPECIAL_TOKENS_DICT 
import clip 
from dataset import build_input_from_story, tokenize, VisualStoryDataset
from model import MultiCaptionGeneratorConfig, MutiCaptionGenerator
from train import build_input 

logger = logging.getLogger(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"  


class CDEScore: 
    """
    calculate the semantic change matchness
    """
    def __init__(self, args, tokenizer):
        self.config = MultiCaptionGeneratorConfig() 
        self.model = MutiCaptionGenerator(self.config, args, tokenizer)
        data = torch.load(args.model_path) 
        self.model.load_state_dict(data['state_dict'])
        self.tokenizer = tokenizer 
        self.model.eval() 
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-9) 
    
    def score(self, input_embs, sentence_index, token_type_ids, attention_mask=None): 
        transformer_outputs_outputs = self.model.captioner(
            inputs_embeds=input_embs, 
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ) 
        hidden_states = transformer_outputs_outputs[0] # （bsz, seq_len, model_d）
        latent_dynamic_hiddents = hidden_states.index_select(1, sentence_index[1:]) # (bsz, num_of_dyn, model_d) 

        latent_output, _ = self.model.latent_dynamic_module(latent_dynamic_hiddents) 
        sentence_hidden_delta = latent_dynamic_hiddents[:, 1:, :] - latent_dynamic_hiddents[:, :-1, :] 
        latent_output_delta = latent_output[:, :-1, :] - latent_dynamic_hiddents[:, :-1, :] 
        x = self.cos(sentence_hidden_delta, latent_output_delta) 
        CDE_score = torch.pow(2, -torch.mean(torch.log(((x+1)/2))))
        return CDE_score.item()



def eval(): 
    parser = ArgumentParser()  
    parser.add_argument("--test_data_path", type=str, default="./dataset/test.json", help="Path of the test dataset") 
    parser.add_argument("--tokenizer_path", type=str, default="ckpt/gpt2", help="Path of the tokenizer") 
    parser.add_argument("--clip_path", type=str, default="ckpt/clip/ViT-B-32.pt", help="Path of the clip model") 
    parser.add_argument("--model_path", type=str, default="ckpt/visual_storytelling/latest.pt", help="Path of the plan-CVAE model") 
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--prefix_size', type=int, default=512)
    parser.add_argument('--prefix_length', type=int, default=10) 
    args = parser.parse_args()
    
    logger.info('Prepare tokenizer and model') 
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)  
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 

    image_encoder, preprocess = clip.load(args.clip_path, device=device) 


    logger.info('Prepare datasets') 
    test_data_list = build_input_from_story(args.test_data_path) 
    test_dataset = VisualStoryDataset(test_data_list, tokenizer, image_encoder, preprocess, device) 

    CDE_scorer = CDEScore(args, tokenizer) 
    logger.info('Begin Evaluation') 

    CDE_score_list = []
    for idx, (image_features_list, history_txt_list, caption_ids) in enumerate(test_dataset): 
        input_embs, _, sentence_index, token_types = build_input(image_features_list, history_txt_list, caption_ids, CDE_scorer.model, args) 
        CDE_score = CDE_scorer.score(input_embs, sentence_index, token_types)
        CDE_score_list.append(CDE_score) 
        break 

    with open('cde_result.txt', 'w', encoding='utf-8') as f: 
        for score in CDE_score_list: 
            f.write(str(score))


if __name__ == "__main__":
    eval()