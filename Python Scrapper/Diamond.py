import re
from collections import Counter
import requests
from bs4 import BeautifulSoup
import mysql.connector
from datetime import datetime, timedelta

# URLs and matching emerald tables
tasks = [
    {
        "url": "https://wijayagems.com/collections/blue-sapphire?pf_opt_shape=Oval",
        "table_delete_insert": "emerald_oval",
        "table_conditional_insert": "emerald_ovald"
    },
    {
        "url": "https://wijayagems.com/collections/blue-sapphire?pf_opt_shape=Round",
        "table_delete_insert": "emerald_round",
        "table_conditional_insert": "emerald_roundd"
    },
    {
        "url": "https://wijayagems.com/collections/blue-sapphire?pf_opt_shape=Rectangle",
        "table_delete_insert": "emerald_princess",
        "table_conditional_insert": "emerald_princessd"
    },
    {
        "url": "https://wijayagems.com/collections/blue-sapphire?pf_opt_shape=Pear",
        "table_delete_insert": "emerald_pear",
        "table_conditional_insert": "emerald_peard"
    }
]

# Connect to the emerald MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # Default WAMP password
    database="emerald"
)

cursor = conn.cursor()

# Process each task
for task in tasks:
    url = task["url"]
    table_main = task["table_delete_insert"]
    table_diff = task["table_conditional_insert"]

    # Fetch page content
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    text = str(soup)

    # Extract weights and prices
    weights = re.findall(r'"name"\s*:\s*"([\d.]+)ct', text)
    prices = re.findall(r'"price"\s*:\s*(\d+)', text)

    weights = [float(w) for w in weights]
    prices = [int(p) for p in prices]
    pairs = list(zip(weights, prices))

    # Keep only unique weights
    weight_counts = Counter(weights)
    unique_pairs = [(w, p) for w, p in pairs if weight_counts[w] == 1]

    # Clear the main table before insert
    cursor.execute(f"DELETE FROM {table_main}")

    for weight, price in unique_pairs:
        cursor.execute(f"INSERT INTO {table_main} (weight, price) VALUES (%s, %s)", (weight, price))

    # Conditional insert to 'd' table if 30 days passed
    cursor.execute(f"SELECT MAX(timestamp) FROM {table_diff}")
    last_timestamp = cursor.fetchone()[0]

    insert_to_diff = False
    if last_timestamp is None:
        insert_to_diff = True
    else:
        now = datetime.now()
        if now - last_timestamp >= timedelta(days=30):
            insert_to_diff = True

    if insert_to_diff:
        for weight, price in unique_pairs:
            cursor.execute(f"INSERT INTO {table_diff} (weight, price) VALUES (%s, %s)", (weight, price))
        print(f"✅ Inserted into {table_diff} (monthly update).")
    else:
        print(f"⏳ Skipped insert into {table_diff} (not 30 days yet).")

# Commit changes and close connections
conn.commit()
cursor.close()
conn.close()

print("✅ All emerald data saved successfully.")
