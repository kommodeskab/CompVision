@echo off
setlocal enabledelayedexpansion

REM Run for AdamW optimizer
REM for /L %%i in (0,1,9) do (
REM     set "dropout=0.%%i"
REM     python .\main.py --image_size=128 --batch_size=64 --name=adamw_dropout_0%%i --dropout_prob=!dropout! --optimizer=adamw
REM )

REM Run for SGD optimizer
for /L %%i in (0,1,9) do (
    set "dropout=0.%%i"
    python .\main.py --image_size=128 --batch_size=64 --name=sgd_dropout_0%%i --dropout_prob=!dropout! --optimizer=sgd
)

pause