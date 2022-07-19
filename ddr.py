import math

from config import DATA_PATH, MODELS_PATH, TARGET_LANGS
import os
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
from scipy.spatial import distance
import fasttext
import googletrans

fasttext.FastText.eprint = lambda x: None


class DDR:
    """
    This class if for constructing DDR features
    """

    def __init__(self, dictionary, embedding_model):
        self.dictionary = dictionary
        self.model = embedding_model


class DDR_aggr(DDR):
    def __init__(self, dictionary, embedding_model):
        super().__init__(dictionary, embedding_model)
        self.categories_embedds = None

    def construct(self, df, text_column):
        self.categories_embedds = self._get_categories_embeddings()

        tqdm.pandas()
        df["DDR"] = df[text_column].progress_apply(self._calculate_ddr)
        df = df.reset_index(drop=True)
        return df

    def _get_categories_embeddings(self):
        """
        Samples from each category of the dictionary words and average
         their word embeddings to represent that category
        :param dict:
        :param model:
        :return: dictionary of category and embedding
        """
        cat_embeds = {}
        for category, words in self.dictionary.items():
            if len(words) < DDR_SAMPLE_SIZE:
                sample_words = words
            else:
                sample_words = random.sample(words, DDR_SAMPLE_SIZE)
            average_embedding = np.mean([self.model[word] for word in sample_words], axis=0)
            cat_embeds[category] = average_embedding
        return cat_embeds

    def _calculate_ddr(self, text):
        """
        compute each text's similarity to each dictionary category and average them.
        :return: return a vector of the size of #categories in dictionary
        """

        embeds = [self.model[word] for word in text.split()]
        text_embed = sum(embeds) / np.linalg.norm(sum(embeds))
        # if not np.isnan(text_embed).any():
        sim_scores = [round(1 - distance.cosine(embed, text_embed), 3) for
                      cat, embed in self.categories_embedds.items()]

        return sim_scores


class DDR_pairwise(DDR):

    def calculate_similarity(self, df, moral_column_name='moral_word', target_column_name='word'):
        tqdm.pandas()
        print("calculating similarity")
        df['sim_score'] = df.progress_apply(self._get_distance, args=(moral_column_name, target_column_name), axis=1)
        return df

    def _get_distance(self, row, word1, word2):
        if str(row[word1]) == 'nan' or str(row[word2]) == 'nan':
            return pd.NA
        else:
            return round(1 - distance.cosine(self.model[row[word1]], self.model[row[word2]]), 3)


if __name__ == '__main__':
    for lang in TARGET_LANGS:
        df = pd.read_csv(os.path.join(DATA_PATH, "joint_df_" + lang + ".csv"), index_col=0)
        print(df.head())
        model = fasttext.load_model(os.path.join(MODELS_PATH,
                                                 'cc.' + googletrans.LANGCODES[lang.lower()] + '.300.bin'))
        print('loaded fasttext model')
        ddr = DDR_pairwise(dictionary=None, embedding_model=model)
        df = ddr.calculate_similarity(df)
        df.to_csv(os.path.join(DATA_PATH, 'df_scores_' + lang + '.csv'))
