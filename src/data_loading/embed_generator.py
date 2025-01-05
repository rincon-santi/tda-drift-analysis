from abc import ABC, abstractmethod
import os
import json
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from src.resource_measurement.resource_measurer import init_measurements, finish_measurements, get_measurements
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec
import pandas as pd

class EmbedGenerator(ABC):
    def __init__(self, name="EmbedGenerator", **kwargs):
        self.name = name
        self.data = None
        self.embeddings = None

    def generate_embeddings(self, data, **kwargs):
        resources = init_measurements()
        self.data = data.copy()
        self.embeddings = self._generate_embeddings(**kwargs)
        resources = finish_measurements(resources)
        return get_measurements(resources)
    
    @abstractmethod
    def _generate_embeddings(self, **kwargs):
        pass

    def save(self, path):
        parameters = {
            "name": self.name
        }
        with open(os.join(path, "parameters.json"), "w") as f:
            json.dump(parameters, f)
        with open(os.join(path, "embeddings.npy"), "wb") as f:
            np.save(f, self.embeddings)
        with open(os.join(path, "data.npy"), "wb") as f:
            np.save(f, self.data)

    @staticmethod
    def load(path):
        with open(os.join(path, "parameters.json"), "r") as f:
            parameters = json.load(f)
        generator = EmbedGenerator(
            name=parameters["name"]
        )
        generator.embeddings = np.load(os.join(path, "embeddings.npy"))
        generator.data = np.load(os.join(path, "data.npy"))
        return generator
    
    def get_embeddings(self):
        return {
            "sentiment": self.data["sentiment"],
            "text": self.data["text"],
            "embeddings": self.embeddings
        }
    
class BertEmbeddingsGenerator(EmbedGenerator):
    def __init__(self, name="BertEmbeddingsGenerator", base_model="bert-base-uncased", max_length=128, batch_size=32):
        super().__init__(name=name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.base_model = base_model
        self.tokenizer = BertTokenizer.from_pretrained(self.base_model)
        self.model = BertModel.from_pretrained(self.base_model)

    def _generate_embeddings(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        self.model.to(device)
        embeddings = []

        for i in range(0, len(self.data["text"]), self.batch_size):
            batch_texts = self.data["text"][i:i + self.batch_size]
            if type(batch_texts) == pd.Series:
                batch_texts = batch_texts.tolist()

            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, 
                                    truncation=True, max_length=self.max_length).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)
    
    def save(self, path):
        super().save(path)
        with open(os.join(path, "model.json"), "w") as f:
            json.dump({"base_model": self.base_model, "max_length": self.max_length, 
                       "batch_size": self.batch_size}, f)
            
    @staticmethod
    def load(path):
        with open(os.join(path, "parameters.json"), "r") as f:
            parameters = json.load(f)
        with open(os.join(path, "model.json"), "r") as f:
            model_parameters = json.load(f)
        generator = BertEmbeddingsGenerator(
            name=parameters["name"],
            base_model=model_parameters["base_model"],
            max_length=model_parameters["max_length"],
            batch_size=model_parameters["batch_size"]
        )
        generator.data = np.load(os.join(path, "data.npy"))
        generator.embeddings = np.load(os.join(path, "embeddings.npy"))
        return generator
    
class W2VEmbeddingsGenerator(EmbedGenerator):
    def __init__(self, name="W2VEmbeddingsGenerator"):
        super().__init__(name=name)
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

    def _generate_embeddings(self):
        tokens = self.data["text"].apply(word_tokenize).tolist()  # Ensure tokens is a list of lists

        w2v_model = Word2Vec(vector_size=100, window=5, min_count=5, workers=4)

        w2v_model.build_vocab(tokens)

        w2v_model.train(tokens, total_examples=len(tokens), epochs=w2v_model.epochs)

        embeddings = np.array([np.mean(
            [w2v_model.wv[token] if token in w2v_model.wv else np.zeros(w2v_model.vector_size) for token in sentence], 
            axis=0) for sentence in tokens])
        
        return embeddings