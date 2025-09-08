#python TestMAB.py -e ttt_lever -rid 0 -nsh 200 -nst 5 -rls 500 -g -cuda "cuda:0" &
python TestMAB.py -e ttt -rid 0 -nsh 100 -nst 5 -rls 500 -g -cuda "cuda:0"&
python TestMAB.py -e ttt -rid 0 -nsh 25 -nst 10 -rls 500 -g -cuda "cuda:0" -match -adv "gae"&
python TestMAB.py -e ttt -rid 0 -nsh 50 -nst 10 -rls 500 -g -cuda "cuda:0" -match -adv "gae"&
python TestMAB.py -e ttt -rid 0 -nsh 25 -nst 5 -rls 500 -g -cuda "cuda:0" -match -adv "gae"&
python TestMAB.py -e ttt -rid 0 -nsh 50 -nst 5 -rls 500 -g -cuda "cuda:0" -match -adv "gae"&
python TestMAB.py -e ttt -rid 0 -nsh 100 -nst 5 -rls 500 -g -cuda "cuda:0" -match -adv "gae"&
#python TestMAB.py -e ttt_lever -rid 0 -nsh 100 -nst 5 -rls 500 -g -cuda "cuda:0" -match -adv "gae"&
#python TestMAB.py -e ttt_lever -rid 0 -nsh 100 -nst 1 -rls 500 -g -cuda "cuda:0" -match -adv "gae"&
#python TestMAB.py -e ttt_lever -rid 0 -nsh 100 -nst 3 -rls 500 -g -cuda "cuda:0" -match -adv "gae"

#python TestMAB.py -e ttt_lever -rid 0 -nsh 200 -nst 5 -rls 500 -g -cuda "cuda:0" -match -adv "monte"
#python TestMAB.py -e ttt_lever -rid 0 -nsh 200 -nst 5 -rls 500 -g -cuda "cuda:0" -match -adv "td"&
#python TestMAB.py -e ttt_lever -rid 0 -nsh 200 -nst 5 -rls 500 -g -cuda "cuda:0" -match -adv "gae" -stubborn
