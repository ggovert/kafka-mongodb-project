from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def insert_data_to_postgresql(path, table_name, db_url):
    try:
        df = pd.read_csv(path)        
        engine = create_engine(db_url)
        
        df.to_sql(table_name, engine, if_exists='append', index=False)
        print(f"Data telah dimasukkan ke tabel {table_name}.")
    except Exception as e:
        print(f"Terjadi kesalahan:\n{e}")
