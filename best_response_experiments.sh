source ../rl/bin/activate &
python TestMAB.py -e overcooked -rid 0 -nsh 1 -nst 1 -rls 10 -g -cuda "cuda:0" -paths "model_paths_overcooked.txt" --om --seed 10 --rl_model "PPO"&
python TestMAB.py -e overcooked -rid 1 -nsh 1 -nst 1 -rls 10 -g -cuda "cuda:0" -paths "model_paths_overcooked.txt" --om --seed 20 --rl_model "PPO"&
wait
python TestMAB.py -e overcooked -rid 2 -nsh 1 -nst 1 -rls 10 -g -cuda "cuda:0" -paths "model_paths_overcooked.txt" --om --seed 30 --rl_model "PPO"&
python TestMAB.py -e overcooked -rid 3 -nsh 1 -nst 1 -rls 10 -g -cuda "cuda:0" -paths "model_paths_overcooked.txt" --om --seed 40 --rl_model "PPO"&
wait
python TestMAB.py -e overcooked -rid 4 -nsh 1 -nst 1 -rls 10 -g -cuda "cuda:0" -paths "model_paths_overcooked.txt" --om --seed 50 --rl_model "PPO"&
python TestMAB.py -e overcooked -rid 0 -nsh 1 -nst 1 -rls 10 -g -cuda "cuda:0" -paths "model_paths_overcooked.txt" --om --seed 10 --rl_model "MDQ"&
wait
python TestMAB.py -e overcooked -rid 1 -nsh 1 -nst 1 -rls 10 -g -cuda "cuda:0" -paths "model_paths_overcooked.txt" --om --seed 20 --rl_model "MDQ"&
python TestMAB.py -e overcooked -rid 2 -nsh 1 -nst 1 -rls 10 -g -cuda "cuda:0" -paths "model_paths_overcooked.txt" --om --seed 30 --rl_model "MDQ"&
wait
python TestMAB.py -e overcooked -rid 3 -nsh 1 -nst 1 -rls 10 -g -cuda "cuda:0" -paths "model_paths_overcooked.txt" --om --seed 40 --rl_model "MDQ"&
python TestMAB.py -e overcooked -rid 4 -nsh 1 -nst 1 -rls 10 -g -cuda "cuda:0" -paths "model_paths_overcooked.txt" --om --seed 50 --rl_model "MDQ"&
wait