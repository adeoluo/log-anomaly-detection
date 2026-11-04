# Records docs/demo/http.gif from a real run: `vhs docs/demo/http.tape` from the repo root,
# after scripts/download_data.sh has fetched the HDFS log into ../.bench-data/log-anomaly-detection.
Output docs/demo/http.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 560
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 18ms

Hide
Type `export PS1='$ ' && uv sync -q --inexact && source .venv/bin/activate && export H=../.bench-data/log-anomaly-detection/hdfs_v1` Enter
Sleep 2s
Type `python -m log_anomaly_detection serve --port 18471 --train $H/HDFS.log --train-lines 300000 --labels $H/preprocessed/anomaly_label.csv --idle-lines 2000 >/dev/null 2>&1 &` Enter
Sleep 2s
Type `batch() { sed -n "$1,$2p" $H/HDFS.log | jq -R . | jq -cs '{lines: ., flush: true}'; }` Enter
Type `until curl -s localhost:18471/ready >/dev/null; do sleep 0.5; done; clear` Enter
Sleep 8s
Show
Type `curl -s localhost:18471/ready | jq -c` Enter
Sleep 1.5s
Type `batch 300001 320000 | curl -s -X POST localhost:18471/ingest -d @- | jq -c '.alerts[:3][] | {kind, key, score, events}'` Enter
Sleep 20s
Type `curl -s localhost:18471/metrics | grep -E '^lad_(alerts|lines_ingested|templates|monitor_sessions_scored)'` Enter
Sleep 4s
Hide
Type `kill %1` Enter
Sleep 1s
