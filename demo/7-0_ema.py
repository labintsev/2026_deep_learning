import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Генерация синтетического временного ряда (тренд + сезонность + шум)
np.random.seed(42)
time = np.arange(100)
trend = 0.5 * time
seasonality = 10 * np.sin(2 * np.pi * time / 12)
noise = np.random.normal(0, 3, size=100)
data = trend + seasonality + noise

# Создаем DataFrame
df = pd.DataFrame({"Actual": data}, index=pd.date_range(start="2026-01-01", periods=100, freq="D"))

# 2. Линейный фильтр (Простое скользящее среднее — SMA)
# Окно сглаживания = 7 дней
window_size = 7
df["SMA_Filtered"] = df["Actual"].rolling(window=window_size, min_periods=1).mean()

# Прогноз методом SMA на 10 шагов вперед (последнее известное значение фильтра)
forecast_steps = 10
sma_forecast_value = df["SMA_Filtered"].iloc[-1]
sma_forecast_idx = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq="D")
sma_forecast = pd.Series(sma_forecast_value, index=sma_forecast_idx)

# 3. Простое экспоненциальное сглаживание (SES)
# Коэффициент сглаживания alpha = 0.3 (чем меньше alpha, тем сильнее сглаживание)
alpha = 0.3
df["EMA_Filtered"] = df["Actual"].ewm(alpha=alpha, adjust=False).mean()

# Прогноз методом EMA на 10 шагов вперед (последнее известное значение сглаживания)
ema_forecast_value = df["EMA_Filtered"].iloc[-1]
ema_forecast_idx = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq="D")
ema_forecast = pd.Series(ema_forecast_value, index=ema_forecast_idx)

# 4. Визуализация результатов
plt.figure(figsize=(14, 7))

# Исторические данные и фильтрация
plt.plot(df.index, df["Actual"], label="Исходный ряд (Actual)", color="gray", alpha=0.5, linestyle="--")
plt.plot(df.index, df["SMA_Filtered"], label=f"Линейный фильтр (SMA, window={window_size})", color="blue", linewidth=2)
plt.plot(df.index, df["EMA_Filtered"], label=f"Экспоненциальное сглаживание (EMA, alpha={alpha})", color="red", linewidth=2)

# Прогноз
plt.plot(sma_forecast.index, sma_forecast, label="Прогноз SMA", color="blue", linestyle=":", linewidth=2)
plt.plot(ema_forecast.index, ema_forecast, label="Прогноз EMA", color="red", linestyle=":", linewidth=2)

# Оформление графика
plt.title("Сравнение Линейного фильтра (SMA) и Экспоненциального сглаживания (EMA)")
plt.xlabel("Дата")
plt.ylabel("Значение")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Вывод последних прогнозных значений в консоль
print("Прогноз на следующие 5 дней:")
forecast_table = pd.DataFrame({
    "Прогноз SMA": sma_forecast.head(5),
    "Прогноз EMA": ema_forecast.head(5)
})
print(forecast_table)
