# this file use for delete target sentence from text
import re

def remove_target(text):
    pattern = re.compile(
        r'Brief Hospital Course:\s*\n{0,2}(.*?)'
        r'(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)'
        r'|Discharge Instructions:\n(.*?)'
        r'(?=Followup Instruction)',
        flags=re.DOTALL)
    return pattern.sub('', text)
