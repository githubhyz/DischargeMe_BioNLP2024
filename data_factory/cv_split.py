# Description: Create input data for DischargeMe dataset depending on token length.

import pandas as pd
from pathlib import Path
from utils import remove_target
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold


def main(root_dir="", seed=0, n_folds=5):
    root_dir = Path(root_dir)
    data_dir = root_dir / "" # e.g. 'data/discharge-me-bionlp-acl24-shared-task-on-streamlining-discharge-documentation-1.2/'  # noqa

    df = None
    for state in ['train', 'valid']:
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

        # target
        discharge_target_df = pd.read_csv(dir / 'discharge_target.csv.gz',
                                          keep_default_na=False)
        assert len(discharge_df) == len(discharge_target_df)
        discharge_target_df['total_target_word_count'] = \
            discharge_target_df['discharge_instructions_word_count'] + \
                discharge_target_df['brief_hospital_course_word_count']

        merge_df = pd.merge(discharge_df, discharge_target_df, on='hadm_id')
        if state == 'train':
            df = merge_df
        else:
            df = pd.concat([df, merge_df])
    df = df.reset_index(drop=True)

    num_bins = int(np.floor(1 + np.log2(len(df))))
    
    df["bins"] = pd.cut(
        df['total_target_word_count'], bins=num_bins, labels=False)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (t_, v_) in enumerate(kf.split(X=df, y=df['bins'].values)):
        df.loc[v_, 'fold'] = fold
    df['fold'] = df['fold'].astype(int)

    for fold in range(n_folds):
        print(f"Fold {fold} bins distribution:")
        print(df[df['fold'] == fold]['bins'].value_counts())
        print()

    df = df.drop("bins", axis=1)

    output_path = data_dir / f'{n_folds}folds_df.csv.gz'

    df.to_csv(output_path, index=False, compression='gzip')    

if __name__ == '__main__':
    import fire
    fire.Fire(main)
