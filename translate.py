import IPython
import googletrans
import pandas as pd
from config import TARGET_LANGS, DATA_PATH
from googletrans import Translator
import os


def translate_word(word, lang):
    translator = Translator()
    try:
        trans = translator.translate(word, dest=lang).text
        return trans
    except Exception as e:
        print(word, e)
        return pd.NA


def trans_to_langs(words, langs):
    df = pd.DataFrame()
    df['English'] = words
    for lang in langs:
        print(lang)
        df[lang] = df['English'].apply(translate_word, args=(lang,))
    return df


def replace_translation(df, replace_col, trans_df, lang):
    df_temp = trans_df.set_index('English')
    trans_dict = df_temp.to_dict()[lang]
    df_new = df.copy()
    df_new[replace_col] = df_new[replace_col].map(trans_dict)
    return df_new



