#python TestMAB.py -e ttt_lever -rid 0 -nsh 200 -nst 5 -rls 500 -g -cuda "cuda:0" &
python TestMAB.py -e overcooked -rid 0 -nsh 2 -nst 10 -rls 500 -g -cuda "cuda:0" -paths "model_paths_overcooked.txt"&
python TestMAB.py -e overcooked -rid 0 -nsh 1 -nst 10 -rls 500 -g -cuda "cuda:0" -match -adv "gae" -paths "model_paths_overcooked.txt"&
python TestMAB.py -e overcooked -rid 0 -nsh 3 -nst 10 -rls 500 -g -cuda "cuda:0" -match -adv "gae" -paths "model_paths_overcooked.txt"
python TestMAB.py -e overcooked -rid 0 -nsh 5 -nst 10 -rls 500 -g -cuda "cuda:0" -match -adv "gae" -paths "model_paths_overcooked.txt"&
python TestMAB.py -e overcooked -rid 0 -nsh 3 -nst 5 -rls 500 -g -cuda "cuda:0" -match -adv "gae" -paths "model_paths_overcooked.txt"&
python TestMAB.py -e overcooked -rid 0 -nsh 3 -nst 15 -rls 500 -g -cuda "cuda:0" -match -adv "gae" -paths "model_paths_overcooked.txt"
python TestMAB.py -e overcooked -rid 0 -nsh 3 -nst 20 -rls 500 -g -cuda "cuda:0" -match -adv "gae" -paths "model_paths_overcooked.txt"&
python TestMAB.py -e overcooked -rid 0 -nsh 5 -nst 25 -rls 500 -g -cuda "cuda:0" -match -adv "gae" -paths "model_paths_overcooked.txt"&
python TestMAB.py -e overcooked -rid 0 -nsh 5 -nst 25 -rls 500 -g -cuda "cuda:0" -match -adv "monte" -paths "model_paths_overcooked.txt"
#python TestMAB.py -e ttt_lever -rid 0 -nsh 100 -nst 5 -rls 500 -g -cuda "cuda:0" -match -adv "gae"&
#python TestMAB.py -e ttt_lever -rid 0 -nsh 100 -nst 1 -rls 500 -g -cuda "cuda:0" -match -adv "gae"&
#python TestMAB.py -e ttt_lever -rid 0 -nsh 100 -nst 3 -rls 500 -g -cuda "cuda:0" -match -adv "gae"

#python TestMAB.py -e ttt_lever -rid 0 -nsh 200 -nst 5 -rls 500 -g -cuda "cuda:0" -match -adv "monte"
#python TestMAB.py -e ttt_lever -rid 0 -nsh 200 -nst 5 -rls 500 -g -cuda "cuda:0" -match -adv "td"&
#python TestMAB.py -e ttt_lever -rid 0 -nsh 200 -nst 5 -rls 500 -g -cuda "cuda:0" -match -adv "gae" -stubborn
