import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Lấy dữ liệu thời tiết từ OpenWeatherMap
def get_weather_data(api_key, city):
    base_url = 'http://api.openweathermap.org/data/2.5/forecast'
    params = {'q': city, 'appid': api_key, 'units': 'metric'}  # Xử dụng đơn vị mét
    response = requests.get(base_url, params=params)
    data = response.json()
    return data

# Chuyển data json thành dataframe
def parse_weather_data(data):
    timestamps = [entry['dt'] for entry in data['list']]
    dates = [datetime.utcfromtimestamp(timestamp) for timestamp in timestamps]
    temperatures = [entry['main']['temp'] for entry in data['list']]
    df = pd.DataFrame({'Date': dates, 'Temperature': temperatures})
    return df

# Dự đoán nhiệt độ bằng Linear Regression
def predict_temperature(df):
    df['Timestamp'] = df['Date']  # Giữ nguyên timestamp
    df['Hour'] = df['Timestamp'].dt.hour % 24  # Lấy giờ từ timestamp và chia lấy dư để giữ giờ từ 0 đến 23
    X = df['Hour'].values.reshape(-1, 1)
    y = df['Temperature'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm thử
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Dự đoán cho ngày tiếp theo mỗi 3 giờ
    tomorrow = datetime.now() + pd.DateOffset(days=1)
    tomorrow_hours = [(tomorrow.hour + i) % 24 for i in range(0, 24, 3)]  
    temperature_predictions = model.predict([[hour] for hour in tomorrow_hours])
    print("Predicted Temperatures for Tomorrow:")
    for i in range(len(tomorrow_hours)):
        print(f'Hour: {tomorrow_hours[i]}, Temperature: {temperature_predictions[i]:.2f} °C')

    # Vẽ biểu đồ sử dụng seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Biểu đồ thực tế
    sns.scatterplot(x=X_test.flatten(), y=y_test, color='black', label='Actual')

    # Biểu đồ dự đoán
    sns.scatterplot(x=tomorrow_hours, y=temperature_predictions, color='red', label='Predicted')

    # Đường hồi quy
    sns.lineplot(x=X_test.flatten(), y=y_pred, color='blue', linewidth=3, label='Regression Line')

    plt.title('Temperature Prediction')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()

# Chạy chương trình
api_key = '1c5bc9e7bb827190882ba4f5f7b60230'
city = 'Tokyo'

weather_data = get_weather_data(api_key, city)
df = parse_weather_data(weather_data)
predict_temperature(df)
