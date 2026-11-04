# Records docs/demo/parse-eval.gif from a real run: `vhs docs/demo/parse-eval.tape` from the repo root.
Output docs/demo/parse-eval.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 640
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q --inexact && source .venv/bin/activate && clear` Enter
Sleep 3s
Show
Type `python -m log_anomaly_detection parse-eval --data-dir tests/fixtures/loghub_2k` Enter
Sleep 6s
