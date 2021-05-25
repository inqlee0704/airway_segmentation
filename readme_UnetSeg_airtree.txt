for s in $(cat VidaCase.in); do (( python inference.py $s )); done
