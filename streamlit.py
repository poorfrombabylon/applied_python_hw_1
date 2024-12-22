import streamlit as st
import datetime
import asyncio
import aiohttp
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

URL_TEMPLATE = "https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={API_KEY}"

def analyze_city(city_data):
    city_data["rolling_mean"] = (
        city_data["temperature"].rolling(window=30, min_periods=1).mean()
    )

    seasonal_stats = (
        city_data.groupby("season")["temperature"].agg(["mean", "std"]).reset_index()
    )
    city_data = city_data.merge(seasonal_stats, on="season", how="left")

    city_data["min"] = city_data["mean"] - 2 * city_data["std"]
    city_data["max"] = city_data["mean"] + 2 * city_data["std"]
    city_data["not_ok"] = ((city_data["temperature"] < city_data["min"]) | (
                city_data["temperature"] > city_data["max"])).astype(int)

    return city_data


def analyze_data_full(df):
    results = []
    for city in df["city"].unique():
        city_data = df[df["city"] == city]
        results.append(analyze_city(city_data))
    return pd.concat(results)


def preprocessing(path):
    df = pd.read_csv(path)
    df = df.sort_values(by=["city", "timestamp"]).reset_index(drop=True)

    df_res = analyze_data_full(df)

    return df, df_res


async def get_weather_async(city, api_key=None):
    url = URL_TEMPLATE.format(
        city=city,
        API_KEY=api_key
    )

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()
            return json.loads(content)


def main():
    uploaded_file = st.file_uploader("Choose CSV-file", type=["csv"])
    if uploaded_file is None:
        st.write("Please upload your CSV-file.")
        return

    df_orig, df_stats = preprocessing(uploaded_file)

    city = st.selectbox("Select the city: ", df_orig["city"].unique())
    api_key = st.text_input("Enter the api key for OpenWeather", None)

    if not api_key is None:
        weather = asyncio.run(get_weather_async(city, api_key))

        if "main" not in weather:
            st.error(weather["message"])
            return

        curr_temp = weather["main"]["temp"]

        normal_mean, normal_std = df_stats.loc[
            (df_stats.city == city) & (df_stats.season == "winter"), ["mean", "std"]
        ].values[0]

        lower_bound = normal_mean - 2 * normal_std
        upper_bound = normal_mean + 2 * normal_std

        is_anomaly = True if curr_temp > upper_bound or curr_temp < lower_bound else False

        if is_anomaly:
            st.error(f"Current temperature in {city}: **{curr_temp} °C**. It is NOT ok")
        else:
            st.success(f"Current temperature in {city}: **{curr_temp} °C**. It is ok.")

        fig, ax = plt.subplots(figsize=(10, 7))
        plt.title(f"Temperature distribution in {city}")
        sns.boxplot(df_stats.loc[df_stats.city == city], y="temperature", x="season", ax=ax)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(15, 8))
        plt.title(f"Historical temperature in {city}")

        df_slice = df_stats.loc[df_stats.city == city]
        df_slice.loc[:, "timestamp"] = pd.to_datetime(df_slice["timestamp"])
        df_anomaly = df_slice.loc[df_slice["is_anomaly"] == 1]

        ax.plot(df_slice["timestamp"], df_slice["temperature"], label="Temperature")
        ax.plot(df_slice["timestamp"], df_slice["min"], label="Normal temperature lower bound",
                linestyle='dashed')
        ax.plot(df_slice["timestamp"], df_slice["max"], label="Normal temperature upper bound",
                linestyle='dashed')
        ax.scatter(df_anomaly["timestamp"], df_anomaly["temperature"], color="red", marker="*", label="anomaly")

        ax.legend()
        st.pyplot(fig)


if __name__ == "__main__":
    st.set_page_config(
        initial_sidebar_state="auto",
        page_title="OpenWeather Demo",
    )

