import torch 
from model import MultiCaptionGeneratorConfig, MutiCaptionGenerator 
from transformers import GPT2Tokenizer 


SPECIAL_TOKENS = ["[bos]", "[eos]", "[dyn]"] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]", 'additional_special_tokens': ["[dyn]",]}



class Config():
    def __init__(self):
        self.max_length = 40
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.top_k = 40
        self.top_p = 0
        self.min_length = 11
        self.no_sample = False
        self.temperature = 0.9
        self.model_checkpoint = "./ckpt/visual_storytelling/latest.pt" 
        self.tokenizer_path = "./ckpt/gpt2"
        self.batch_size = 2
        self.beam_size = 10
        self.penalty = 0.1 
        self.prefix_size = 512 
        self.prefix_length = 10 


def main(): 
    args = Config() 

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)  
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
    model_config = MultiCaptionGeneratorConfig() 
    model = MutiCaptionGenerator(model_config, args, tokenizer)  
    
    weight_dict = torch.load(args.model_checkpoint, map_location=None) 
    model.load_state_dict(weight_dict['state_dict'], strict=False)



if __name__ == "__main__":
    main()