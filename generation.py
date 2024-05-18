import pandas as pd
import torch
from prompt import input_extract_all, prompt_dict_instructs, prompt_dict_brief
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from tqdm import tqdm


def generate_summary(df_input, model_name, target, device="cuda"):
    if target == "brief_hospital_course":
        prompt_for_input = prompt_dict_brief
        max_target_length = 832

    elif target == "discharge_instructions":
        prompt_for_input = prompt_dict_instructs
        max_target_length = 792

    df_input = pd.read_csv(df_input, compression='gzip', header=0, sep=',', quotechar='"')

    input_text_extract = df_input["text_without_target"].apply(lambda x: input_extract_all(x, prompt_for_input))

    tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=True, sep_token="<sep>")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, from_flax=True, torch_dtype=torch.float16)
    model.to(f"cuda:{device}")

    def generate_text(input_text):
        input = tokenizer(
            "summarize: " + input_text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=1596, 
            add_special_tokens=True, 
        ).to(model.device)

        input_ids = input.input_ids

        set_seed(42)
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_target_length, 
            min_length=10,
            num_beams=4, 
            do_sample=True,
            diversity_penalty=0,
            length_penalty=1.1,
            no_repeat_ngram_size=4,
            early_stopping=True
        )
        summary_txt = tokenizer.decode(outputs.squeeze(), skip_special_tokens=True)
        torch.cuda.empty_cache()        
        return summary_txt
    
    summaries = []
    for text in tqdm(input_text_extract):
        summaries.append(generate_text(text))
    
    return summaries