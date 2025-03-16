%%writefile stopwordRemoval.py
from util import *

# Import necessary libraries
import nltk
from nltk.corpus import stopwords

class StopwordRemoval():

    def fromList(self, text):
        """
        Stopword Removal from Tokenized Sentences

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """
        nltk.download('stopwords')  # Ensure stopwords are available
        stop_words = set(stopwords.words('english'))

        removed_txt = [[word for word in sent if word.lower() not in stop_words] for sent in text]

        return removed_txt
