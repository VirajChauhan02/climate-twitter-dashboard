@echo off
cd /d "%~dp0"
python -m pip install --upgrade pip
python -m pip install streamlit pandas scikit-learn plotly nltk
python -m streamlit run "%~dp0Climate_Change_Twitter_Dashboard.py"
pause
