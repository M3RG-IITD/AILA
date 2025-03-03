@echo off
call conda activate afmai
cd "..\app"
streamlit run app.py
