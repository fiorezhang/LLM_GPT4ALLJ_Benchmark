@echo off

for /l %%i in (8,4,20) do (
    python gpt4all-j-benchmark.py -n 32 -t %%i -o threadTest.csv
    timeout /t 30
)