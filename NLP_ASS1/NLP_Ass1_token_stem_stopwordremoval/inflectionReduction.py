import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

class InflectionReduction:

    def __init__(self):
        # Download the WordNet dataset if not already downloaded
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        # Initiablize the stemmer and lemmatizer
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def reduce(self, text, method='lemmatization'):
        """
        Perform stemming or lemmatization on a list of tokenized sentences

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence
        method : str, optional
            The method to use for inflection reduction. Options are 'stemming' or 'lemmatization'.
            Default is 'lemmatization'.

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of
            stemmed/lemmatized tokens representing a sentence
        """

        rd_txt = []

        # Iterate through each sentence
        for sentence in text:
            rd_sent = []
            # Iterate through each token in the sentence
            for token in sentence:
                if method == 'stemming':
                    # Apply stemming
                    token = self.stemmer.stem(token)
                elif method == 'lemmatization':
                    # Apply lemmatization
                    token = self.lemmatizer.lemmatize(token)
                else:
                    raise ValueError("Invalid method. Choose 'stemming' or 'lemmatization'.")
                rd_sent.append(token)
            rd_txt.append(rd_sent)

        return rd_txt