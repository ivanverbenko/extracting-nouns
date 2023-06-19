import re
import time

import torch
import pymorphy2
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForTokenClassification
import nltk

class RuPosTagger:
    
    def __init__(self, model_name="KoichiYasuoka/bert-base-russian-upos"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.morph = pymorphy2.MorphAnalyzer()
        self.MAX_WORDS = 200
        nltk.download('stopwords')

    def get_nouns(self, text):
        """ разбивает текст на подстроки с помощью split_text.
        Для каждой подстроки ищет всех существительные в этой подстроке"""
        preprocessed_text = self.__preprocess_text(text)
        substrings = self.__split_text(preprocessed_text)
        nouns = []
        for substring in substrings:
            batch_nouns = self.__get_nouns_from_substring(substring)
            nouns.extend(batch_nouns)
        return nouns
    
    def __get_nouns_from_substring(self, substring):
        """Получает на вход подстроку текста (не более 200 слов),
        вычисляет POS-теги для каждого слова, фильтрует существительные"""
        tokens = self.tokenizer.encode(substring, return_tensors="pt",max_length=512,truncation='longest_first')
        outputs = self.model(tokens)
        # Получение POS-тегов из предсказанных меток
        predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze()
        predicted_tags = [self.model.config.id2label[label_id.item()] for label_id in predicted_labels]
        nouns = []
        for i, tag in enumerate(predicted_tags[1:-1]): # Пропускаем первый и последний токены
            if tag == "NOUN":
                word = self.tokenizer.decode([tokens[0][i + 1]])
                nouns.append(word)
        return nouns
    
    def __preprocess_text(self, text):
        """Предобработка текста."""
        text = text.lower()
        text = re.sub('[^а-я-]+', ' ', text)
        text = re.sub(' +', ' ', text)
        lemmas = [self.morph.parse(word)[0].normal_form for word in set(text.split())]
        filtered_lemmas = [word for word in lemmas if word not in stopwords.words('russian')]
        return ' '.join(set(filtered_lemmas))

    def __split_text(self, text):
        """
        Разбивает текст на подстроки длиной до `max_words` слов.
        Иначе тензоры получаются >512 и приходится отбрасывать
        """
        words = text.split()
        return [" ".join(words[i:i+self.MAX_WORDS]) for i in range(0, len(words), self.MAX_WORDS)]

