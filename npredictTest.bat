@echo off

for /l %%i in (128,128,512) do (
    python gpt4all-j-benchmark.py -n %%i -t 24 -o nPredictTest.csv
    timeout /t 30
)