from PIL import Image
import os
import json
from torch.utils.data import Dataset
import torch
import random
import numpy as np
from transformers import CLIPImageProcessor, CLIPProcessor, CLIPModel
from torchvision import transforms

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=False):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    # print('tests------------', type(prompt_embeds), prompt_embeds.keys())  tests------------ 
    # <class 'transformers.modeling_outputs.BaseModelOutputWithPooling'> odict_keys(['last_hidden_state', 'pooler_output'])
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_promptsXL(text_encoders, tokenizers, prompts):
    prompt_embeds_all = []
    pooled_prompt_embeds_all = []

    for prompt in prompts:
        prompt_embeds, pooled_prompt_embeds = encode_promptXL(text_encoders, tokenizers, prompt)
        prompt_embeds_all.append(prompt_embeds)
        pooled_prompt_embeds_all.append(pooled_prompt_embeds)

    return torch.stack(prompt_embeds_all), torch.stack(pooled_prompt_embeds_all)

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_promptXL(text_encoders, tokenizers, prompt):
    prompt_embeds_list = []

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

# Adapted from examples.dreambooth.train_dreambooth_lora_sdxl
# Here, we compute not just the text embeddings but also the additional embeddings
# needed for the SD XL UNet to operate.
def compute_embeddings_for_prompts(prompts, text_encoders, tokenizers):
    with torch.no_grad():
        prompt_embeds_all, pooled_prompt_embeds_all = encode_promptsXL(text_encoders, tokenizers, prompts)
        add_text_embeds_all = pooled_prompt_embeds_all

        prompt_embeds_all = prompt_embeds_all.to(accelerator.device)
        add_text_embeds_all = add_text_embeds_all.to(accelerator.device)
    return prompt_embeds_all, add_text_embeds_all


class LoRA_Json_Dataset(Dataset):
    def __init__(self, json_file, metadata_path, transform=None, tokenizer=None, text_encoder=None, plip_image_processor=None, plip_model=None, use_expand=False):
        self.data = []
        self.transform = transform
        self.metadata_path = metadata_path
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.json_file = os.path.join(metadata_path, json_file)
        # self.clip_image_processor = CLIPImageProcessor()
        self.plip_image_processor = plip_image_processor  #CLIPProcessor.from_pretrained("vinid/plip")
        self.plip_model = plip_model  #CLIPModel.from_pretrained("vinid/plip")
        self.i_drop_rate = 0.05
        self.ti_drop_rate = 0.05
        self.t_drop_rate = 0.05
        self.use_expand = use_expand

        with open(self.json_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.data.append(json.loads(line))

        # print(type(self.data),type(self.data[1]))
        # print(self.data[1]['input_image'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图像路径和对应的 prompt
        # print(idx)
        input_name = self.data[idx]['input_image']
        prompt = self.data[idx]['edit_prompt']
        edit_name = self.data[idx]['edited_image']   
        # 使用 PIL 读取图像
        input_image = Image.open(os.path.join(self.metadata_path, input_name)).convert('RGB')
        edit_image = Image.open(os.path.join(self.metadata_path, edit_name)).convert('RGB')

        # random crop
        # win_size = 512
        # w = input_image.size[0]
        # if w > win_size:
        #     cor_xa, cor_ya = random.randint(0, w - win_size), random.randint(0, w - win_size)
        #     input_image = input_image.crop((cor_xa, cor_ya, cor_xa+win_size, cor_ya+ win_size))
        #     edit_image = edit_image.crop((cor_xa, cor_ya, cor_xa+ win_size, cor_ya+ win_size))

        img_embeds = []


        # 应用变换（如果定义了的话）
        if self.transform:
            # edit_image = 2 * (np.array(edit_image) / 255) - 1   # [-1,1]
            input_image = self.transform(input_image).to(torch.float16).cuda()
            edit_image = self.transform(edit_image).to(torch.float16).cuda()

        # Encode prompt
        if self.tokenizer and self.text_encoder:
            text_inputs = tokenize_prompt(self.tokenizer, prompt, tokenizer_max_length=None)
            text_embedding = encode_prompt(
                self.text_encoder,
                text_inputs.input_ids,
                text_inputs.attention_mask,
                text_encoder_use_attention_mask=False
            )
            # print(text_embedding.size())  #torch.Size([1, 77, 768])
            text_embedding = text_embedding.squeeze(dim=0)


        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # 返回图像和 prompt
        sample = {'original_pixel_values': input_image, 
                  'edited_pixel_values': edit_image, 
                  'input_ids': text_embedding, 
                  'image_embeds': img_embeds,
                  "drop_image_embed": drop_image_embed}
        return sample
    


class LoRA_Json_DatasetXL(Dataset):
    def __init__(self, json_file, metadata_path, transform=None, tokenizer=None, text_encoder=None, plip_image_processor=None, plip_model=None):
        self.data = []
        self.transform = transform
        self.metadata_path = metadata_path
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.json_file = os.path.join(metadata_path, json_file)
        self.clip_image_processor = CLIPImageProcessor()
        self.plip_image_processor = plip_image_processor  #CLIPProcessor.from_pretrained("vinid/plip")
        self.plip_model = plip_model  #CLIPModel.from_pretrained("vinid/plip")
        self.i_drop_rate = 0.05
        self.ti_drop_rate = 0.05
        self.t_drop_rate = 0.05

        with open(self.json_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.data.append(json.loads(line))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图像路径和对应的 prompt
        # print(idx)
        input_name = self.data[idx]['input_image']
        prompt = self.data[idx]['edit_prompt']
        edit_name = self.data[idx]['edited_image']   
        # 使用 PIL 读取图像
        input_image = Image.open(os.path.join(self.metadata_path, input_name))
        edit_image = Image.open(os.path.join(self.metadata_path, edit_name))

        # random crop
        win_size = 512
        w = input_image.size[0]
        if w > win_size:
            cor_xa, cor_ya = random.randint(0, w - win_size), random.randint(0, w - win_size)
            input_image = input_image.crop((cor_xa, cor_ya, cor_xa+win_size, cor_ya+ win_size))
            edit_image = edit_image.crop((cor_xa, cor_ya, cor_xa+ win_size, cor_ya+ win_size))

        clip_image = self.clip_image_processor(images=input_image, return_tensors="pt")   #.pixel_values

        # 应用变换（如果定义了的话）
        if self.transform:
            input_image = self.transform(input_image.convert('RGB')).to(torch.float16).cuda()
            edit_image = self.transform(edit_image.convert('RGB')).to(torch.float16).cuda()

        inputs = self.plip_image_processor(text=prompt, images=input_image, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].to().cuda()
        inputs['attention_mask'] = inputs['attention_mask'].to(torch.float16).cuda()
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16).cuda()
        

        outputs = self.plip_model(**inputs)
        img_embeds = outputs.vision_model_output[0]
        # print(img_embeds.size())
        img_embeds = img_embeds.squeeze(dim=0)

        # Encode prompt
        if self.tokenizer and self.text_encoder:
            # text_inputs = tokenize_prompt(self.tokenizer, prompt, tokenizer_max_length=None)
            # text_embedding = encode_prompt(
            #     self.text_encoder,
            #     text_inputs.input_ids,
            #     text_inputs.attention_mask,
            #     text_encoder_use_attention_mask=False
            # )
            # text_embedding = text_embedding.squeeze(dim=0)

            prompt_embeds_all, pooled_prompt_embeds_all = encode_promptXL(self.text_encoder, self.tokenizer, prompt)
            # print(prompt_embeds_all.size(), pooled_prompt_embeds_all.size())  #torch.Size([1, 77, 2048]) torch.Size([1, 1280])

        # concat condition img and prompt
        # text_embedding = torch.cat([text_embedding, img_embeds], dim=0)


        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # 返回图像和 prompt
        sample = {'original_pixel_values': input_image, 
                  'edited_pixel_values': edit_image, 
                  'input_ids': prompt_embeds_all, 
                  'prompt_embeds': 1,
                  'add_text_embeds': 1,
                  'clip_image':clip_image,
                  "drop_image_embed": drop_image_embed}
        return sample
