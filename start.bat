@echo off
call C:\ProgramData\anaconda3\Scripts\activate.bat C:\Users\user\.conda\envs\def_env
set PYTHONPATH=C:\Users\user\.conda\envs\def_env\python.exe
cd /d "%~dp0"
code .