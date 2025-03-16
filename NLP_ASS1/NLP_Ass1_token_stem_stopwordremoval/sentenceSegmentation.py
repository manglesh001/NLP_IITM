from util import *

import re
import nltk
from nltk.tokenize import sent_tokenize

class SentenceSegmentation():

    def naive(self, text):
        """
        Sentence Segmentation using a Naive Approach

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """
        sgmt_txt = re.split(r'[.!?]+\s*', text.strip())  
        sgmt_txt = [sentence for sentence in sgmt_txt if sentence]        # Remove empty strings 


        return sgmt_txt

    def punkt(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """
        nltk.download('punkt')  # Check Punkt tokenizer is available
        sgmt_txt = sent_tokenize(text)

        return sgmt_txt
