import numpy as np
import pandas as pd
from config import DATA_PATH, TARGET_LANGS
import os
from translate_df import translate_word


def explode_df(df, columns=["moral_found", 'moral_word'], explode_col='moral_word'):
    df = df.to_dict(orient='list')
    df = pd.DataFrame(list(df.items()), columns=columns)
    df = df.explode(explode_col)
    df.reset_index(drop=True, inplace=True)
    return df


def merge_dfs(df1, df2):
    df1['tmp'] = 1
    df2['tmp'] = 1
    join_df = pd.merge(df1, df2, on='tmp')
    join_df.drop(columns=['tmp'], inplace=True)
    return join_df


def create_joint_df(df, lang, translate=True):
    """
    Prepare joint df for the language
    """
    if translate:
        translator = np.vectorize(translate_word)
        new_df = pd.DataFrame()
        for label in df.columns:
            new_df[label] = translator(df[label], lang)
    else:
        new_df = df

    wealt_words = new_df[['Inequality', 'Wealth']]
    mfd = new_df.drop(columns=['Inequality', 'Wealth'])

    wealt_df = explode_df(wealt_words, columns=["word_cat", 'word'], explode_col='word')
    mfd_df = explode_df(mfd)
    joint_df = merge_dfs(wealt_df, mfd_df)
    return joint_df


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(DATA_PATH, 'mfd_wealth_V2.csv'))
    for lang in TARGET_LANGS:
        print("\n\n", lang, "\n\n")
        joint_df = create_joint_df(df, lang=lang)
        print(joint_df.head(4))
        joint_df.to_csv(os.path.join(DATA_PATH, "joint_df_" + lang + ".csv"))
