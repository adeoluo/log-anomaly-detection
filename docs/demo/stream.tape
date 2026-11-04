# Records docs/demo/stream.gif from a real run: `vhs docs/demo/stream.tape` from the repo root,
# after scripts/download_data.sh has fetched the HDFS log into ../.bench-data/log-anomaly-detection.
Output docs/demo/stream.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 560
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 18ms

Hide
Type `export PS1='$ ' && uv sync -q --inexact && source .venv/bin/activate && export H=../.bench-data/log-anomaly-detection/hdfs_v1 && clear` Enter
Sleep 3s
Show
Type `python -m log_anomaly_detection stream --train $H/HDFS.log --train-lines 1000000 --labels $H/preprocessed/anomaly_label.csv \` Enter
Type `  --input $H/HDFS.log --skip-lines 1000000 --max-lines 200000 --show 10` Enter
Sleep 32s
