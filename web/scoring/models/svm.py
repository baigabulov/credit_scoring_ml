import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

class SVMModel:
    def __init__(self):
        self.model = SVC(
            kernel='rbf',  # радиальная базисная функция
            C=1.0,  # параметр регуляризации
            gamma='scale',  # коэффициент ядра
            probability=True,  # включение вероятностных оценок
            random_state=42
        )
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data, is_training=True):
        """
        Предобработка данных: кодирование категориальных признаков и масштабирование числовых
        """
        df = data.copy()
        
        # Кодируем категориальные признаки
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column])
        
        # Масштабируем числовые признаки
        if is_training:
            df = pd.DataFrame(
                self.scaler.fit_transform(df),
                columns=df.columns
            )
        else:
            df = pd.DataFrame(
                self.scaler.transform(df),
                columns=df.columns
            )
            
        return df
    
    def train(self, X, y):
        """
        Обучение модели
        """
        # Предобработка данных
        X_processed = self.preprocess_data(X, is_training=True)
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Обучение модели
        self.model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Точность модели: {accuracy:.2f}")
        print("\nОтчет о классификации:")
        print(report)
        
        return accuracy, report
    
    def predict(self, X):
        """
        Предсказание для новых данных
        """
        X_processed = self.preprocess_data(X, is_training=False)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """
        Предсказание вероятностей классов
        """
        X_processed = self.preprocess_data(X, is_training=False)
        return self.model.predict_proba(X_processed)
    
    def predict_dict(self, data: dict) -> float:
        """
        Предсказание для данных в формате словаря
        """
        # Преобразуем словарь в DataFrame
        df = pd.DataFrame([data])
        
        # Предобработка данных
        X_processed = self.preprocess_data(df, is_training=False)
        
        # Получаем предсказание
        prediction = self.model.predict(X_processed)[0]
        
        return float(prediction)
    
    def save_model(self, path):
        """
        Сохранение модели, энкодеров и скейлера
        """
        model_path = os.path.join(path, 'svm_model.joblib')
        encoders_path = os.path.join(path, 'svm_label_encoders.joblib')
        scaler_path = os.path.join(path, 'svm_scaler.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoders, encoders_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Модель сохранена в {model_path}")
        print(f"Энкодеры сохранены в {encoders_path}")
        print(f"Скейлер сохранен в {scaler_path}")
    
    @classmethod
    def load_model(cls, path):
        """
        Загрузка сохраненной модели
        """
        model_path = os.path.join(path, 'svm_model.joblib')
        encoders_path = os.path.join(path, 'svm_label_encoders.joblib')
        scaler_path = os.path.join(path, 'svm_scaler.joblib')
        
        instance = cls()
        instance.model = joblib.load(model_path)
        instance.label_encoders = joblib.load(encoders_path)
        instance.scaler = joblib.load(scaler_path)
        
        return instance
