from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import os
import json
import logging
from src.resource_measurement.resource_measurer import init_measurements, finish_measurements, get_measurements

LOGGER = logging.getLogger(__name__)

class DriftGenerator(ABC):
    def __init__(self, name="DriftGenerator", desired_drift_proportion=0.1, **kwargs):
        self.name = name
        self.desired_drift_proportion = desired_drift_proportion
        self.no_drift_data = None
        self.drift_data = None

    def generate_drift(self, data, **kwargs):
        resources = init_measurements()
        result = self._generate_drift(data, **kwargs)
        resources = finish_measurements(resources)
        return result, get_measurements(resources)

    @abstractmethod
    def _generate_drift(self, data, **kwargs):
        pass

    def get_mixed_data(self, no_drift_proportion=0.5):
        no_drift_data = self.no_drift_data.sample(frac=no_drift_proportion)
        drift_data = self.drift_data.sample(frac=1 - no_drift_proportion)
        all_data = pd.concat([no_drift_data, drift_data])
        all_data = all_data.sample(frac=1)
        return all_data
    
    def plot_distributions(self):
        df = pd.concat([self.no_drift_data.assign(group='standard'), 
                        self.drift_data.assign(group='drift')], axis=0)
        sentiment_distribution = df.groupby('group')['sentiment'].value_counts(normalize=True).unstack()
        sentiment_distribution.plot(
            kind='bar', figsize=(10, 6), 
            title=f"Comparación de Distribuciones de Sentimiento en Grupos Standard y Drift ({self.name})", 
            xlabel="Grupo", ylabel="Proporción de Sentimientos", legend=True, grid=True)


    def save(self, path):
        parameters = {
            "name": self.name,
            "desired_drift_proportion": self.desired_drift_proportion
        }
        with open(os.join(path, "parameters.json"), "w") as f:
            json.dump(parameters, f)
        with open(os.join(path, "no_drift_data.csv"), "w") as f:
            self.no_drift_data.to_csv(f)
        with open(os.join(path, "drift_data.csv"), "w") as f:
            self.drift_data.to_csv(f)

    @staticmethod
    def load(path):
        with open(os.join(path, "parameters.json"), "r") as f:
            parameters = json.load(f)
        generator = TemporalDriftGenerator(
            name=parameters["name"],
            desired_drift_proportion=parameters["desired_drift_proportion"]
        )
        generator.no_drift_data = pd.read_csv(os.join(path, "no_drift_data.csv"))
        generator.drift_data = pd.read_csv(os.join(path, "drift_data.csv"))
        return generator


class TemporalDriftGenerator(DriftGenerator):
    def __init__(self, name="TemporalDriftGenerator", desired_drift_proportion=0.1):
        super().__init__(name=name, desired_drift_proportion=desired_drift_proportion)
        self.drift_start = None

    def _generate_drift(self, data):
        drift_df = data.copy()
        drift_df['date'] = pd.to_datetime(drift_df['date'], errors='coerce')
    
        drift_df = drift_df.dropna(subset=['date'])
        drift_df = drift_df.sort_values(by='date')

        drift_df['month'] = drift_df['date'].dt.to_period('M')
        drift_df['month'] = drift_df['month'].astype(str)

        drift_df['month'].value_counts().sort_index()
    
        self.drift_start = drift_df['month'].value_counts().sort_index().index[-len(drift_df['month'].value_counts()) // (10 - int(self.desired_drift_proportion * 10))]

        self.drift_data = drift_df.loc[drift_df['month'] >= self.drift_start]
        self.no_drift_data = drift_df.loc[drift_df['month'] < self.drift_start]

        return self
    
    def save(self, path):
        parameters = {
            "name": self.name,
            "desired_drift_proportion": self.desired_drift_proportion,
            "drift_start": self.drift_start
        }
        with open(os.join(path, "parameters.json"), "w") as f:
            json.dump(parameters, f)
        with open(os.join(path, "no_drift_data.csv"), "w") as f:
            self.no_drift_data.to_csv(f)
        with open(os.join(path, "drift_data.csv"), "w") as f:
            self.drift_data.to_csv(f)

    @staticmethod
    def load(path):
        with open(os.join(path, "parameters.json"), "r") as f:
            parameters = json.load(f)
        generator = TemporalDriftGenerator(
            name=parameters["name"],
            desired_drift_proportion=parameters["desired_drift_proportion"]
        )
        generator.no_drift_data = pd.read_csv(os.join(path, "no_drift_data.csv"))
        generator.drift_data = pd.read_csv(os.join(path, "drift_data.csv"))
        generator.drift_start = parameters["drift_start"]
        return generator
    

class SelectionDriftGenerator(DriftGenerator):
    def __init__(self, column_name, criteria, name="DriftGenerator", desired_drift_proportion=0.1):
        super().__init__(name, desired_drift_proportion)
        self.column_name = column_name
        self.criteria = criteria
        self.threshold = None

    def _generate_drift(self, data):
        df = data.copy()
        df["filter_criteria"] = df[self.column_name].apply(self.criteria)

        self.threshold = df["filter_criteria"].quantile(self.desired_drift_proportion)
        print(f"Umbral de longitud (percentil {self.desired_drift_proportion}): {self.threshold}")

        self.drift_data = df.loc[df["filter_criteria"] > self.threshold]
        self.no_drift_data = df.loc[df["filter_criteria"] <= self.threshold]

        return self
    
    def save(self, path):
        parameters = {
            "name": self.name,
            "desired_drift_proportion": self.desired_drift_proportion,
            "column_name": self.column_name,
            "criteria": self.criteria,
            "threshold": self.threshold
        }
        with open(os.join(path, "parameters.json"), "w") as f:
            json.dump(parameters, f)
        with open(os.join(path, "no_drift_data.csv"), "w") as f:
            self.no_drift_data.to_csv(f)
        with open(os.join(path, "drift_data.csv"), "w") as f:
            self.drift_data.to_csv(f)

    @staticmethod
    def load(path):
        with open(os.join(path, "parameters.json"), "r") as f:
            parameters = json.load(f)
        generator = SelectionDriftGenerator(
            column_name=parameters["column_name"],
            criteria=parameters["criteria"],
            name=parameters["name"],
            desired_drift_proportion=parameters["desired_drift_proportion"]
        )
        generator.no_drift_data = pd.read_csv(os.join(path, "no_drift_data.csv"))
        generator.drift_data = pd.read_csv(os.join(path, "drift_data.csv"))
        generator.threshold = parameters["threshold"]
        LOGGER.warning(f"Selection based drift generator loaded from file. It may not be usable for new drift generation. Modify criteria if needed.")

        return generator
    

class NoiseDriftGenerator(DriftGenerator):
    def __init__(self, name="NoiseDriftGenerator", desired_drift_proportion=0.1):
        super().__init__(name=name, desired_drift_proportion=desired_drift_proportion)
        
    def _generate_drift(self, data):
        df = data.copy()
        output_length=int(len(df)//(1-self.desired_drift_proportion))
        print(f"Generando ruido para {output_length} registros desde {len(df)} registros originales")
        new_records = pd.DataFrame(columns=['sentiment', 'text', 'date'])
        new_records['sentiment'] = np.random.randint(0, 2, output_length-len(df))
        new_records['text'] = [''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz '), 50)) for _ in range(output_length-len(df))]
        allowed_dates = pd.date_range(start=df['date'].min(), 
                                      periods=min(output_length-len(df), 1000),
                                      freq='H')
        new_records['date'] = np.random.choice(allowed_dates, output_length-len(df))
        self.no_drift_data = df
        self.drift_data = new_records