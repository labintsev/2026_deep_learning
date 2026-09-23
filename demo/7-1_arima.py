import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1. Генерация синтетического ряда с выраженным трендом и сезонностью (12 месяцев)
np.random.seed(42)
dates = pd.date_range(start="2018-01-01", periods=84, freq="ME") # 7 лет данных

# Компоненты: линейный тренд + сильная годовая сезонность + случайный шум
trend = 0.4 * np.arange(84)
seasonality = 15 * np.sin(2 * np.pi * np.arange(84) / 12)
noise = np.random.normal(0, 3, size=84)
series = 100 + trend + seasonality + noise

df = pd.DataFrame({"Actual": series}, index=dates)

# Разделяем на обучающую выборку (первые 6 лет) и тестовую для проверки (последний год)
train = df.iloc[:-12]
test = df.iloc[-12:]

# 2. Обучение модели ARIMA (не учитывает сезонность напрямую)
# Параметры (p, d, q): order=(p, d, q)
# p - порядок авторегрессии, d - порядок интегрирования (разностей), q - порядок скользящего среднего
arima_model = ARIMA(train["Actual"], order=(2, 1, 1))
arima_result = arima_model.fit()

# Прогноз ARIMA на 12 месяцев вперед
arima_forecast = arima_result.forecast(steps=12)

# 3. Обучение модели SARIMA (учитывает и тренд, и сезонность)
# order=(p, d, q) - несезонные параметры
# seasonal_order=(P, D, Q, S) - сезонные параметры, где S=12 (период сезонности — 12 месяцев)
sarima_model = SARIMAX(train["Actual"], 
                       order=(1, 1, 1), 
                       seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False) # disp=False убирает лишний вывод оптимизатора

# Прогноз SARIMA на 12 месяцев вперед
sarima_forecast = sarima_result.forecast(steps=12)

# 4. Визуализация результатов
plt.figure(figsize=(14, 7))

# Исторические данные (Обучающая выборка)
plt.plot(train.index, train["Actual"], label="История (Train)", color="black", linewidth=1.5)

# Реальные данные за последний год (Тестовая выборка)
plt.plot(test.index, test["Actual"], label="Реальные данные (Test)", color="green", linewidth=2)

# Прогнозы моделей
plt.plot(test.index, arima_forecast, label="Прогноз ARIMA", color="red", linestyle="--", linewidth=2)
plt.plot(test.index, sarima_forecast, label="Прогноз SARIMA", color="blue", linestyle="-.", linewidth=2)

# Оформление
plt.title("Сравнение прогнозов моделей ARIMA и SARIMA")
plt.xlabel("Дата")
plt.ylabel("Значения")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# Вывод метрик для сравнения точности (MAE - средняя абсолютная ошибка)
mae_arima = np.mean(np.abs(test["Actual"] - arima_forecast))
mae_sarima = np.mean(np.abs(test["Actual"] - sarima_forecast))

print(f"Средняя абсолютная ошибка (MAE) ARIMA: {mae_arima:.2f}")
print(f"Средняя абсолютная ошибка (MAE) SARIMA: {mae_sarima:.2f}")
