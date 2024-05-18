import re

ITEM_SIMPLE = {
    "Sex", 
    "Service", 
    "Allergies", 
    "Chief Complaint", 
    "Major Surgical or Invasive Procedure"
}

ITEM_COMPLEX = {
    "History of Present Illness", 
    "Past Medical History", 
    "Pertinent Results",
    "Medications on Admission", 
    "Discharge Medications", 
    "Discharge Disposition", 
    "Discharge Diagnosis", 
    "Discharge Condition"
}

prompt_dict_brief = {
    "Sex": "Gender details are as follows: ",
    "Chief Complaint": "The primary reason for the visit is summarized as follows: ",
    "History of Present Illness": "An overview of the current illness's history is provided as follows: ",
    "Pertinent Results": "Clinically significant findings impacting the treatment and diagnosis are as follows: ",
    "Past Medical History": "A summary of the patient's past medical history is as follows: ",
    "Allergies": "Information on any allergies is detailed as follows: ",
    "Major Surgical or Invasive Procedure": "Details on any major surgeries or invasive procedures are as follows: ",
    "Service": "The service details are provided as follows: ",
}

prompt_dict_instructs = {
    "Sex": "Gender details are as follows: ",
    "Chief Complaint": "The primary reason for the visit is summarized as follows: ",
    "History of Present Illness": "An overview of the current illness's history is provided as follows: ",
    "Past Medical History": "A summary of the patient's past medical history is as follows: ",
    "Allergies": "Information on any allergies is detailed as follows: ",
    "Major Surgical or Invasive Procedure": "Details on any major surgeries or invasive procedures are as follows: ",
    "Medications on Admission": "Medications upon admission are detailed as follows: ",
    "Service": "The service details are provided as follows: ",
    "Discharge Diagnosis": "The final diagnosis at discharge is as follows: ",
    "Discharge Disposition": "The disposition at discharge is provided as follows: ",
    "Discharge Condition": "The patient's condition upon discharge is described as follows: ",
    "Discharge Medications": "Medications prescribed at discharge are as follows: ",
}

def input_extract(text, item):
    patterns = {
        'Sex': r'Sex:   (\w+)\n',
        'Service': r'Service: (\w+)\n',
        'Allergies': r'Allergies:\s*\n(.*?)(?=\n\s*[A-Z_]+[^\n:]+:)',
        'Chief Complaint': r'Chief Complaint:\s*\n(.*?)(?=\n\s*[A-Z_]+[^\n:]+:)',
        'Major Surgical or Invasive Procedure': r'Major Surgical or Invasive Procedure:\s*\n(.*?)(?=\n\s*[A-Z_]+[^\n:]+:)',
        'History of Present Illness': r'History of Present Illness:\s*\n(.*?)(?=Past Medical History:)',
        'Past Medical History': r'Past Medical History:\s*\n(.*?)(?=Social History:)',
        'Pertinent Results': r'Pertinent Results:\s*\n(.*?)(?=Medications on Admission:|Discharge Medications:|Discharge Disposition:|Discharge Diagnosis:|Discharge Condition:|Followup Instructions:)',
        'Medications on Admission': r'Medications on Admission:\s*\n(.*?)(?=Discharge Medications:)',
        'Discharge Medications': r'Discharge Medications:\s*\n(.*?)(?=Discharge Disposition:)',
        'Discharge Disposition': r'Discharge Disposition:\s*\n(.*?)(?=Discharge Diagnosis:)',
        'Discharge Diagnosis': r'Discharge Diagnosis:\s*\n(.*?)(?=Discharge Condition:)',
        'Discharge Condition': r'Discharge Condition:\s*\n(.*?)(?=Followup Instructions:)',
    }
    
    pattern = patterns.get(item)

    if not pattern:
        return "Unknown."
    
    match = re.search(pattern, text, flags=re.DOTALL)

    if not match:
        return "Unknown."
    
    match_text = match.group(1)

    if not match_text:
        return "Unknown."
    else:
        if item in ITEM_SIMPLE:
            if item == "Sex":
                if match_text == 'M':
                    return "Male"
                elif match_text == 'F':
                    return "Female"
            return match_text

        elif item in ITEM_COMPLEX:
            if item == 'Pertinent Results':
                match_text = re.sub(r'^.*\b\d{1,2}:\d{2}\s*(AM|PM|am|pm).*$', '', match_text, flags=re.MULTILINE)

            part_texts = []
            for part_text in match_text.split('  \n'):
                part_part_texts = []
                for part_part_text in part_text.split('\n'):
                    part_part_text = part_part_text.strip()

                    if not re.search(r'[a-zA-Z0-9]', part_part_text):
                        continue

                    if re.match(r'^.*:', part_part_text) and len(part_part_texts) > 0:
                        if part_part_texts[-1][-1] != '.' and part_part_texts[-1][-1] != ':':
                            part_part_texts[-1] += '.'

                    if  item == "Pertinent Results" or \
                            item == 'Past Medical History' or \
                            item == 'Medications on Admission' or \
                            item == 'Discharge Medications' or \
                            item == 'Discharge Disposition' or \
                            item == 'Discharge Diagnosis':
                        if part_part_text[:len('-')] == '-':
                            part_part_text = part_part_text[len('-'):]

                        if item == "Discharge Medications" or item == "Medications on Admission" or item == "Pertinent Results":
                            if re.match(r'^\d+\.\s', part_part_text):
                                if part_part_text[-1] != '.':
                                    part_part_text += '.'
                                #part_part_text = re.sub(r'^(\d+)\.\s', r'Step \1. ', part_part_text)
                                part_part_text = re.sub(r'^(\d+)\.\s', r'* ', part_part_text)
                        else:
                            # number plus dot plus space such as '1. '
                            if re.match(r'^\d+\.\s', part_part_text):
                                part_part_text = part_part_text[re.search(r'\d+\.\s', part_part_text).end():]

                    part_part_texts.append(part_part_text)

                if len(part_part_texts) == 0:
                    continue

                part_text = ' '.join(part_part_texts)

                if part_text[-1] == ':':
                    if item == "Discharge Disposition": # ie: "Home With Service. Facility:" -> "Home With Service."
                        part_text = part_text.split('.')[0]
                    else:
                        continue
                    
                if part_text[-1] != '.':
                    part_text += '.'

                part_texts.append(part_text)

            return re.sub(r'\s{2,}', ' ', ' '.join(part_texts)).strip()
        

def input_extract_all(text, prompt):
    aggregated_info = "The patient's name is provided as follows: ___."
    for item in prompt:
        extract = input_extract(text, item)

        if item == "Discharge Condition" and ":" in extract:
            extract = " is".join(extract.split(":"))
        if "\n" in extract:
            extract = " ".join(extract.split("\n"))
        aggregated_info += "<sep>" + prompt[item] + extract
        if not aggregated_info.endswith('.'):
            aggregated_info += '.'

    return aggregated_info

def target_extract(text):
    part_texts = re.split(r'\n\s*\n', text)
    part_texts = [part_text.strip() for part_text in part_texts]
    part_texts = [part_text for part_text in part_texts if part_text]
    part_texts = [re.sub(r'\s{2,}', ' ', part_text) for part_text in part_texts]
    return '\n\n'.join(part_texts).strip()