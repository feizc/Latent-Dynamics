import torch 
import torch.nn.functional as F 
import logging 
import copy 
import numpy as np
from model import MultiCaptionGeneratorConfig, MutiCaptionGenerator 
from transformers import GPT2Tokenizer 
from utils import SPECIAL_TOKENS, SPECIAL_TOKENS_DICT 
from dataset import build_input_from_story, VisualStoryDataset, tokenize
import clip 
from train import build_input 


logger = logging.getLogger(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"  



def greedy_decode(image_features_list, history_txt_list, model, tokenizer, args): 
    bos, eos, dyn, img, txt = [x[0] for x in tokenize(SPECIAL_TOKENS, tokenizer)] 
    ys = [bos] 
    for i in range(args.max_length): 
        tokens = torch.Tensor(ys).long().unsqueeze(0).to(device) 
        input_embs, _, sentence_index, token_types = build_input(image_features_list, history_txt_list, tokens, model, args) 
        logits = model.step(input_embs, sentence_index, token_types) 
        logits = logits.cpu().data.numpy() 
        next_word = np.argsort(logits[0])[-1] 
        ys.append(next_word) 
        if next_word == eos: 
            break
    return ys 



def beam_search(image_features_list, history_txt_list, model, tokenizer, args): 
    bos, eos, dyn, img, txt = [x[0] for x in tokenize(SPECIAL_TOKENS, tokenizer)] 
    current_output = [bos] 
    hyplist = [([], 0., current_output)] 
    comp_hyplist = [] 
    best_state = None

    for i in range(args.max_length): 
        new_hyplist = []
        argmin = 0 
        for out, lp, st in hyplist: 
            tokens = torch.Tensor(out).long().unsqueeze(0).to(device) 
            input_embs, _, sentence_index, token_types = build_input(image_features_list, history_txt_list, tokens, model, args) 
            logits = model.step(input_embs, sentence_index, token_types) 
            logp = F.log_softmax(logits, dim=-1) 
            lp_vec = logp.cpu().data.numpy() + lp 
            lp_vec = np.squeeze(lp_vec) 
            if i >= args.min_length: 
                new_lp = lp_vec[eos] + args.penalty * (len(out) + 1) 
                comp_hyplist.append((out, new_lp)) 
                if best_state is None or best_state < new_lp:
                    best_state = new_lp 
            count = 1
            for o in np.argsort(lp_vec)[::-1]: 
                if o == tokenizer.unk_token_id or o == eos:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == args.beam_size:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = copy.deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = copy.deepcopy(st)
                    new_st.append(int(o))
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == args.beam_size:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                count += 1
        hyplist = new_hyplist 
    
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1]
        return maxhyps
    else:
        return [([], 0)]




class GenerateConfig():
    def __init__(self):
        self.max_length = 20
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.top_k = 40
        self.top_p = 0
        self.min_length = 11
        self.no_sample = False
        self.temperature = 0.9
        self.model_checkpoint = "./ckpt/visual_storytelling/latest.pt" 
        self.tokenizer_path = "./ckpt/gpt2"
        self.clip_path = './ckpt/clip/ViT-B-32.pt' 
        self.test_data_path = './dataset/test.json' 
        self.decode_strategy = 'beam_search'
        self.batch_size = 1
        self.beam_size = 5
        self.penalty = 0.1 
        self.prefix_size = 512 
        self.prefix_length = 10 



def main(): 
    args = GenerateConfig() 

    logger.info('Prepare tokenizer and model') 
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)  
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 

    image_encoder, preprocess = clip.load(args.clip_path, device=device) 

    model_config = MultiCaptionGeneratorConfig() 
    model = MutiCaptionGenerator(model_config, args, tokenizer)  
    
    weight_dict = torch.load(args.model_checkpoint, map_location=None) 
    model.load_state_dict(weight_dict['state_dict'], strict=False) 

    logger.info('Prepare dataset') 
    test_data_list = build_input_from_story(args.test_data_path) 
    test_dataset = VisualStoryDataset(test_data_list, tokenizer, image_encoder, preprocess, device) 

    logger.info('Begin to generate') 
    model.eval() 
    generate_list = []
    with torch.no_grad(): 
        for idx, (image_features_list, history_txt_list, _) in enumerate(test_dataset): 
            if args.decode_strategy == 'greedy':
                caps = greedy_decode(image_features_list, history_txt_list, model, tokenizer, args) 
            elif args.decode_strategy == 'beam_search': 
                caps = beam_search(image_features_list, history_txt_list, model, tokenizer, args)[0][0]
            sentence = tokenizer.decode(caps) 
            generate_list.append(sentence)

    with open('story_result.txt', 'w', encoding='utf-8') as f: 
        for l in generate_list: 
            f.write(l + '\n')


if __name__ == "__main__":
    main()
    