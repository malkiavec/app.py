import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
import sklearn
import requests
import time
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# API Endpoint
API_URL = "https://data.ny.gov/resource/6nbc-h7bj.json"
CSV_FILE = "lottery_data.csv"

# Function to fetch data from API and store in CSV
@st.cache_data
def fetch_and_store_data():
    try:
        response = requests.get(API_URL)
        data = response.json()
        df = pd.DataFrame(data)

        # Extract relevant columns (modify based on API response structure)
        df = df[['draw_date', 'winning_numbers']]
        df['draw_date'] = pd.to_datetime(df['draw_date'])
        df = df.sort_values(by='draw_date', ascending=True)

        # Split winning numbers into separate columns
        df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']] = df['winning_numbers'].str.split(" ", expand=True).astype(int)

        # Save to CSV
        df.to_csv(CSV_FILE, index=False)
        return df

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Load data
df = fetch_and_store_data()

# Sidebar settings
st.sidebar.header("Mutation Adjustment")
mutation_level = st.sidebar.slider("Mutation Strength", 0.0, 1.0, 0.5)

st.sidebar.subheader("XGBoost Filtering")
use_xgboost = st.sidebar.checkbox("Enable XGBoost Filtering", True)

# Train XGBoost classifier
def train_xgboost(df):
    X = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']]
    X['Valid'] = 1  # Assume all past draws are valid
    y = X.pop('Valid')

    model = XGBClassifier()
    model.fit(X, y)
    return model

if use_xgboost:
    xgb_model = train_xgboost(df)

# Generate random mutations
def generate_mutations(draw, mutation_level):
    mutated_draw = draw.copy()
    for _ in range(int(mutation_level * len(draw))):
        mutated_draw[random.randint(0, len(draw) - 1)] = random.randint(1, 59)
    return sorted(mutated_draw)

# LSTM model for sequence learning
def build_lstm():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(6, 1)),
        Dense(6, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

lstm_model = build_lstm()

# Generate predictions
st.subheader("Generated Predictions")
selected_draw = random.sample(range(1, 60), 6)
st.write("Base Draw:", selected_draw)

mutated_draws = [generate_mutations(selected_draw, mutation_level) for _ in range(10)]
if use_xgboost:
    filtered_draws = [draw for draw in mutated_draws if xgb_model.predict([draw])[0] == 1]
else:
    filtered_draws = mutated_draws

st.write("Filtered Predictions:", filtered_draws)

# Transition Graph
st.subheader("Transition Graph")
G = nx.DiGraph()
for draw in filtered_draws:
    for i in range(len(draw) - 1):
        G.add_edge(draw[i], draw[i + 1])

plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
st.pyplot(plt)

# Heatmap
st.subheader("Number Transition Heatmap")
transition_matrix = np.zeros((59, 59))

for draw in filtered_draws:
    for i in range(len(draw) - 1):
        transition_matrix[draw[i] - 1, draw[i + 1] - 1] += 1

sns.heatmap(transition_matrix, cmap="coolwarm", linewidths=0.5)
st.pyplot(plt)

st.write("Mutation optimization completed!")
