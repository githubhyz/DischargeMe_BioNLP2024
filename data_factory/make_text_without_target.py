# Description: This script is used to remove target from discharge text for test_phase_2 data.

import pandas as pd
from pathlib import Path
from utils import remove_target

def main(root_dir='/working', seed=0, n_folds=5):
    root_dir = Path(root_dir)
    data_dir = root_dir / 'data/discharge-me-bionlp-acl24-shared-task-on-streamlining-discharge-documentation-1.2/'  # noqa

    df = None
    for state in ['test_phase_2',]:
        dir = data_dir / state
        # input text
        discharge_df = pd.read_csv(dir / 'discharge.csv.gz',
                                   keep_default_na=False)
        texts = discharge_df['text']
        remove_target_texts = []
        for text in texts:
            text = remove_target(text)
            remove_target_texts.append(text)
        discharge_df['text_without_target'] = remove_target_texts
        discharge_df = discharge_df.drop(columns=['text'])

        output_path = dir / f'discharge_without_target.csv.gz'
        discharge_df.to_csv(output_path, index=False, compression='gzip')    

if __name__ == '__main__':
    import fire
    fire.Fire(main)
