@echo off
call conda activate afmai
cd "C:\Users\Admin\Desktop\Automation\AFMBench\Data"
streamlit run afm_data_json.py
