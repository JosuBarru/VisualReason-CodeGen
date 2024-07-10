# import os
# import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
# os.environ['CODEX_QUANTIZED'] = '1'
# os.environ['LOAD_MODELS'] = '0'
# os.environ['DATASET'] = 'refcoco'
# os.environ['EXEC_MODE'] = 'codex'
# script_dir = os.path.abspath('/gaueko0/users/eamor002/viper')
# sys.path.append(script_dir)
# from src.main_simple_lib import *

# my_dataset = datasets.get_dataset(config.dataset)
# query = my_dataset.__getitem__(0)['query']
# #show_single_image(img)
# code = get_code(query)
# print(code)


text = '\n    # Return the pizza\n    image_patch = ImagePatch(image)\n    pizza_patches = image_patch.find("pizza")\n    pizza_patches.sort(key=lambda pizza: pizza.horizontal_center)\n    pizza_patch = pizza_patches[0]\n    # Remember: return the pizza\n    return pizza_patch'
text1 = '\n    # Return the pizza\n    image_patch = ImagePatch(image)\n    pizza_patches = image_patch.find("pizza")\n    pizza_patches.sort(key=lambda pizza: pizza.horizontal_center)\n    pizza_patch = pizza_patches[0]'

textGQA = '\n   image_patch = ImagePatch(image)\n   spoon_patches = image_patch.find(""spoon"")\n   spoon_patch = spoon_patches[0]\n   cheese_patches = image_patch.find(""cheese"")\n      for cheese_patch in cheese_patches:\n           if cheese_patch.horizontal_center > spoon_patch.horizontal_center:\n                return bool_to_yesno(cheese_patch.verify_property(""cheese"", ""small and round""))' 
def complete_code(text):
    code = text.split('\n')
    text_= text
    last_line = code[-1]
    tabulate_list = ['      return','           return','               return']
    if 'return' in last_line:
        for line in tabulate_list:
            if  line in last_line:
                text_+='\n'
                text_+='    return None'
                return text_
    else:
        text_+='\n'
        text_+='    return None'
        return text_
    return text

print(complete_code(text))
print(complete_code(text1))
print(complete_code(textGQA))

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# print(torch.cuda.device_count())
# device = torch.device("cuda:0")

# with open('prompts/benchmarks/refcoco.prompt') as f:
#     prompt = f.read().strip()
# p = "Drink with zero alcohol"
# if isinstance(prompt,str):
#     extended_prompt = prompt.replace("INSERT_QUERY_HERE", p)
# #print(extended_prompt)

# quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
# tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-hf', max_length=15000, device_map="auto")
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = 'left'
# model = AutoModelForCausalLM.from_pretrained(
#     'codellama/CodeLlama-7b-hf', 
#     quantization_config = quantization_config,)

# # # modelo = torch.nn.parallel(model, device_ids=[0,1],dim=0)
# start = time.time()
# input_ids = tokenizer(extended_prompt, return_tensors="pt", padding=True, truncation=True)
# input_ids = input_ids["input_ids"].to("cuda")
# generated_ids = model.generate(input_ids, max_new_tokens=128)
# generated_ids = generated_ids[:, input_ids.shape[-1]:]
# generated_text = [tokenizer.decode(gen_id, skip_special_tokens=False) for gen_id in generated_ids]
# generated_text = [text.split('\n\n')[0] for text in generated_text]
# end = time.time() - start
# print(generated_text)
# print(end)

# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   model = torch.nn.DataParallel(model)

# print(model.to("cuda"))