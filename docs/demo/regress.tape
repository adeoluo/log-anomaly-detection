# Records docs/demo/regress.gif from a real run: `vhs docs/demo/regress.tape` from the repo root.
Output docs/demo/regress.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 420
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q --inexact && source .venv/bin/activate && clear` Enter
Sleep 3s
Show
Type `head -4 examples/latency.csv` Enter
Sleep 1.5s
Type `python -m log_anomaly_detection regress examples/latency.csv --column p95_ms --direction both` Enter
Sleep 3s
Type `python -m log_anomaly_detection regress examples/latency.csv --column p95_ms --method bocpd --direction up` Enter
Sleep 4s
