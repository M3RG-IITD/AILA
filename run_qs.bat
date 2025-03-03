@echo off
call conda activate afmai
cd "..\Data"
streamlit run afm_data_json.py
