import torch 
import json 
from torch.utils.data import Dataset  
from PIL import Image
import requests 


SPECIAL_TOKENS = ["[bos]", "[eos]", "[dyn]"] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]", 'additional_special_tokens': ["[dyn]",]}



def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n, o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)



class VisualStoryDataset(Dataset): 
    def __init__(self, data_list, tokenizer, image_encoder, preprocess, device): 
        self.data_list = data_list 
        self.tokenizer = tokenizer 
        self.image_encoder = image_encoder 
        self.preprocess = preprocess 
        self.device = device 
        self.bos, self.eos, self.dyn = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    
    def __len__(self): 
        return len(self.data_list) 
        
    def __getitem__(self, index): 
        story_pair = self.data_list[index] 
        history_images = story_pair['history'][0] 
        history_txt = story_pair['history'][1] 
        caption = story_pair['caption']
        image_features_list = [] 
        for url in history_images: 
            image = Image.open(requests.get(url, stream=True).raw)
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad(): 
                image_features = self.image_encoder.encode_image(image)
            image_features_list.append(image_features) 
        

        history_txt_list = []
        for txt in history_txt: 
            txt_ids = torch.Tensor([self.bos] + tokenize(txt, self.tokenizer) + [self.eos, self.dyn]).long().unsqueeze(0)
            history_txt_list.append(txt_ids) 

        caption_ids = torch.Tensor([self.bos] + tokenize(caption, self.tokenizer) + [self.eos, self.dyn]).long().unsqueeze(0)
        return image_features_list, history_txt_list, caption_ids



def data_preprocess(data_path): 
    with open(data_path, 'r', encoding='utf-8') as f: 
        data_dict = json.load(f) 
    image_url_dict = {} 
    for item in data_dict['images']: 
        image_id = item['id'] 
        for key in item.keys(): 
            if 'url' in key:
                image_url_dict[image_id] = item[key]
    
    story_dict = {}
    for item in data_dict['annotations']: 
        item = item[0]
        story_id = item['story_id'] 
        if story_id not in story_dict.keys(): 
            story_dict[story_id] = {}
            story_dict[story_id]['url'] = [] 
            story_dict[story_id]['txt'] = [] 
        image_id = item['photo_flickr_id'] 
        if image_id not in image_url_dict.keys():
            continue 
        story_dict[story_id]['url'].append(image_url_dict[image_id]) 
        story_dict[story_id]['txt'].append(item['text']) 
    
    with open('./dataset/train.json', 'w', encoding='utf-8') as f: 
        json.dump(story_dict, f, indent=4)



def build_input_from_story(data_path): 
    with open(data_path, 'r', encoding='utf-8') as f: 
        data_dict = json.load(f) 
    
    data_list = []
    for story_id in data_dict.keys(): 
        url_list = data_dict[story_id]['url'] 
        txt_list = data_dict[story_id]['txt'] 
        for i in range(1, len(url_list)): 
            item = {} 
            item['history'] = [] 
            item['history'].append(url_list[:i+1]) 
            item['history'].append(txt_list[:i]) 
            item['caption'] = txt_list[i] 
            data_list.append(item) 
        
    return data_list[:100] 




if __name__ == "__main__":
    data_path = './dataset/VIST/sis/val.story-in-sequence.json' 
    data_preprocess(data_path)