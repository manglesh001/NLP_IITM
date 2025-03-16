from util import *

# Import necessary libraries
import re
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer

class Tokenization():

    def naive(self, text):
        """
        Tokenization using a Naive Approach

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """
        tokenizedText = [re.findall(r'\b\w+\b', sentence) for sentence in text]

        return tokenizedText

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """
        nltk.download('punkt')  # Check Punkt tokenizer is available
        treebank_tokenizer = TreebankWordTokenizer()
        tokenizedText = [treebank_tokenizer.tokenize(sentence) for sentence in text]

        return tokenizedText
