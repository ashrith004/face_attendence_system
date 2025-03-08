import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os  # For file path handling

# Get current date and time
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

# Import autorefresh
from streamlit_autorefresh import st_autorefresh

# Auto-refresh counter
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

# FizzBuzz logic
if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

# File path for attendance CSV
file_path = f"Attendance/Attendance_{date}.csv"

# Check if the file exists
if os.path.exists(file_path):
    try:
        # Read and display the CSV file
        df = pd.read_csv(file_path)
        st.dataframe(df.style.highlight_max(axis=0))
    except Exception as e:
        # Handle any errors while reading the file
        st.error(f"Error reading file: {e}")
else:
    # Display a warning if the file does not exist
    st.warning(f"File not found: {file_path}. Please ensure the file exists.")
