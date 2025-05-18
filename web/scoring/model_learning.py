import pandas as pd

from scoring.models.decision_tree import DecisionTreeModel
from scoring.models.svm import SVMModel
from scoring.models.linear_regression import LinearRegressionModel


LEARNING_COLUMN = 'approved_amount'


class DecisionTreeTraining:

    def __init__(self) -> None:
        self.parent_path = 'scoring'
        self.model = None

    def train_and_save(self):
        # Создание и обучение модели
        parent_path = self.parent_path

        self.model = DecisionTreeModel()
        data = pd.read_csv(parent_path + '/data/data.csv')
        X = data.drop(LEARNING_COLUMN, axis=1)
        y = data[LEARNING_COLUMN]
        accuracy, report = self.model.train(X, y)

        # Сохранение модели
        self.model.save_model(parent_path + '/models/trained/')

    def load_and_predict(self):
        # Загрузка модели
        parent_path = self.parent_path
        self.model = DecisionTreeModel.load_model(parent_path + '/models/trained/')

    def predict(self, data: dict) -> float:
        """
        Предсказание суммы одобренного кредита для данных в формате словаря
        """
        if self.model is None:
            self.load_and_predict()
        return self.model.predict_dict(data)


class SVMTraining:

    def __init__(self) -> None:
        self.parent_path = 'scoring'
        self.model = None

    def train_and_save(self):
        # Загрузка данных
        data = pd.read_csv(self.parent_path + '/data/data.csv')
        X = data.drop(LEARNING_COLUMN, axis=1)
        y = data[LEARNING_COLUMN]
        
        # Создание и обучение модели
        self.model = SVMModel()
        accuracy, report = self.model.train(X, y)
        
        # Сохранение модели
        self.model.save_model(self.parent_path + '/models/trained/')
        
    def load_and_predict(self):
        self.model = SVMModel.load_model(self.parent_path + '/models/trained/')

    def predict(self, data: dict) -> float:
        """
        Предсказание суммы одобренного кредита для данных в формате словаря
        """
        if self.model is None:
            self.load_and_predict()
        return self.model.predict_dict(data)


class LinearRegressionTraining:

    def __init__(self) -> None:
        self.parent_path = 'scoring'
        self.model = None

    def train_and_save(self):
        # Загрузка данных
        data = pd.read_csv(self.parent_path + '/data/data.csv')
        X = data.drop(LEARNING_COLUMN, axis=1)
        y = data[LEARNING_COLUMN]
        
        # Создание и обучение модели
        self.model = LinearRegressionModel()
        metrics = self.model.train(X, y)
        
        # Получение важности признаков
        feature_importance = self.model.get_feature_importance()
        
        # Сохранение модели
        self.model.save_model(self.parent_path + '/models/trained/')
        
    def load_and_predict(self):
        self.model = LinearRegressionModel.load_model(self.parent_path + '/models/trained/')

    def predict(self, data: dict) -> float:
        """
        Предсказание суммы одобренного кредита для данных в формате словаря
        """
        if self.model is None:
            self.load_and_predict()
        return self.model.predict_dict(data)
    