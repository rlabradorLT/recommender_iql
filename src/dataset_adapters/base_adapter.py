# src/dataset_adapters/base_adapter.py

from abc import ABC, abstractmethod
import pandas as pd


class DatasetAdapter(ABC):

    @abstractmethod
    def load_events(self, cfg) -> pd.DataFrame:
        """
        Carga y transforma el dataset raw.

        Debe devolver un DataFrame con las columnas
        estandarizadas para el pipeline:

            user_id
            item_id
            timestamp
            reward

        El adapter es responsable de:

        - cargar raw data
        - limpiar datos
        - mapear ids
        - calcular reward
        """
        pass