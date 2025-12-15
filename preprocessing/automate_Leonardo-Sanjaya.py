# Import Library
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Dataset
df = pd.read_csv('used_car_price_dataset_extended_raw.csv')

# Data Preprocessing
df = df.drop_duplicates()

# Ganti dari "NaN" jadi "No History" (field asli berisikan "None")
df['service_history'] = df['service_history'].fillna('No History')

scaler = StandardScaler()
numerical_cols = ['make_year', 'mileage_kmpl', 'engine_cc', 'owner_count', 'price_usd']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Pilih fitur yang berkorelasi tinggi dengan target
df_selected = df[['make_year', 'mileage_kmpl', 'engine_cc', 'owner_count', 'price_usd']]
df_selected.to_csv('preprocessing/used_car_price_dataset_extended_preprocessing.csv', index=False)