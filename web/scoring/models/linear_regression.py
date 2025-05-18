import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression(
            fit_intercept=True,  # использовать ли свободный член
            n_jobs=-1  # использовать все доступные ядра
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
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
        print(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.2f}")
        print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
        print(f"Коэффициент детерминации (R²): {r2:.2f}")
        
        # Вывод коэффициентов
        coefficients = pd.DataFrame({
            'Признак': X.columns,
            'Коэффициент': self.model.coef_
        })
        print("\nКоэффициенты модели:")
        print(coefficients.sort_values('Коэффициент', ascending=False))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'coefficients': coefficients
        }
    
    def predict(self, X):
        """
        Предсказание для новых данных
        """
        X_processed = self.preprocess_data(X, is_training=False)
        return self.model.predict(X_processed)
    
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
    
    def get_feature_importance(self):
        """
        Получение важности признаков
        """
        importance = pd.DataFrame({
            'Признак': list(self.label_encoders.keys()) + [col for col in self.model.feature_names_in_ if col not in self.label_encoders],
            'Важность': np.abs(self.model.coef_)
        })
        return importance.sort_values('Важность', ascending=False)
    
    def save_model(self, path):
        """
        Сохранение модели, энкодеров и скейлера
        """
        model_path = os.path.join(path, 'linear_regression_model.joblib')
        encoders_path = os.path.join(path, 'linear_regression_encoders.joblib')
        scaler_path = os.path.join(path, 'linear_regression_scaler.joblib')
        
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
        model_path = os.path.join(path, 'linear_regression_model.joblib')
        encoders_path = os.path.join(path, 'linear_regression_encoders.joblib')
        scaler_path = os.path.join(path, 'linear_regression_scaler.joblib')
        
        instance = cls()
        instance.model = joblib.load(model_path)
        instance.label_encoders = joblib.load(encoders_path)
        instance.scaler = joblib.load(scaler_path)
        
        return instance
