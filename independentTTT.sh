# python run_exp.py -e ttt -a DQ -rid 0 -rls 100000 -cuda "cuda:0" &
# python run_exp.py -e ttt -a DQ -rid 1 -rls 100000 -cuda "cuda:0"

# python run_exp.py -e ttt -a DQ -rid 2 -rls 100000 -cuda "cuda:0" &
# python run_exp.py -e ttt -a DQ -rid 3 -rls 100000 -cuda "cuda:0"

# python run_exp.py -e ttt -a DQ -rid 4 -rls 100000 -cuda "cuda:0"&
# python run_exp.py -e ttt -a SDQ -rid 0 -rls 100000 -cuda "cuda:0"

# python run_exp.py -e ttt -a SDQ -rid 1 -rls 100000 -cuda "cuda:0" &
# python run_exp.py -e ttt -a SDQ -rid 2 -rls 100000 -cuda "cuda:0"

# python run_exp.py -e ttt -a SDQ -rid 3 -rls 100000 -cuda "cuda:0" &
# python run_exp.py -e ttt -a SDQ -rid 4 -rls 100000 -cuda "cuda:0"

python run_exp.py -e ttt -a MDQ -rid 0 -rls 100000 -cuda "cuda:0" &
python run_exp.py -e ttt -a MDQ -rid 1 -rls 100000 -cuda "cuda:0"

python run_exp.py -e ttt -a MDQ -rid 2 -rls 100000 -cuda "cuda:0" &
python run_exp.py -e ttt -a MDQ -rid 3 -rls 100000 -cuda "cuda:0"

python run_exp.py -e ttt -a MDQ -rid 4 -rls 100000 -cuda "cuda:0" &
python run_exp.py -e ttt -a PPO -rid 0 -rls 100000 -cuda "cuda:0"

python run_exp.py -e ttt -a PPO -rid 1 -rls 100000 -cuda "cuda:0" &
python run_exp.py -e ttt -a PPO -rid 2 -rls 100000 -cuda "cuda:0"

python run_exp.py -e ttt -a PPO -rid 3 -rls 100000 -cuda "cuda:0" &
python run_exp.py -e ttt -a PPO -rid 4 -rls 100000 -cuda "cuda:0"
