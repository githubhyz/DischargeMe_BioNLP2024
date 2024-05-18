# Description: Check if the input text contains the following strings:

from pathlib import Path
import pandas as pd
from tqdm import tqdm

root_dir = Path('/working')
data_dir = root_dir / 'data/discharge-me-bionlp-acl24-shared-task-on-streamlining-discharge-documentation-1.2/'  # noqa

for csv_path in ['5folds_df.csv.gz', 'test_phase_1/discharge_without_target.csv.gz']:
    df_5folds = pd.read_csv(data_dir / csv_path,
                            keep_default_na=False)

    input_texts = df_5folds["text_without_target"].tolist()
    for input_text in tqdm(input_texts):
        assert 'Brief Hospital Course:' not in input_text
        assert 'Discharge Instructions:' not in input_text
