source ../rl/bin/activate &
#python TestMAB.py -e overcooked -rid 0 -nsh 1 -nst 1 -rls 10 -g -cuda "cpu" -paths "model_paths_overcooked.txt" --pbr --seed 10 --rl_model "PPO"&
#python TestMAB.py -e overcooked -rid 1 -nsh 1 -nst 1 -rls 10 -g -cuda "cpu" -paths "model_paths_overcooked.txt" --pbr --seed 20 --rl_model "PPO"&
#wait
#python TestMAB.py -e overcooked -rid 2 -nsh 1 -nst 1 -rls 10 -g -cuda "cpu" -paths "model_paths_overcooked.txt" --pbr --seed 30 --rl_model "PPO"&
#python TestMAB.py -e overcooked -rid 3 -nsh 1 -nst 1 -rls 10 -g -cuda "cpu" -paths "model_paths_overcooked.txt" --pbr --seed 40 --rl_model "PPO"&
#wait
#python TestMAB.py -e overcooked -rid 4 -nsh 1 -nst 1 -rls 10 -g -cuda "cpu" -paths "model_paths_overcooked.txt" --pbr --seed 50 --rl_model "PPO"&
python TestMAB.py -e overcooked -rid 0 -nsh 1 -nst 1 -rls 10 -g -cuda "cpu" -paths "model_paths_overcooked.txt" --pbr --seed 10 --rl_model "MDQ"&
wait
python TestMAB.py -e overcooked -rid 1 -nsh 1 -nst 1 -rls 10 -g -cuda "cpu" -paths "model_paths_overcooked.txt" --pbr --seed 20 --rl_model "MDQ"&
python TestMAB.py -e overcooked -rid 2 -nsh 1 -nst 1 -rls 10 -g -cuda "cpu" -paths "model_paths_overcooked.txt" --pbr --seed 30 --rl_model "MDQ"&
wait
python TestMAB.py -e overcooked -rid 3 -nsh 1 -nst 1 -rls 10 -g -cuda "cpu" -paths "model_paths_overcooked.txt" --pbr --seed 40 --rl_model "MDQ"&
python TestMAB.py -e overcooked -rid 4 -nsh 1 -nst 1 -rls 10 -g -cuda "cpu" -paths "model_paths_overcooked.txt" --pbr --seed 50 --rl_model "MDQ"&
wait