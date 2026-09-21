python run.py --dataset MKG-Y --srank 128 --vrank 128 --trank 128 --gate_dim 128 --alpha_max 0.2 --optimizer AdamW --learning_rate 1e-3 --batch_size 500 --regularizer wN3 --reg 7e-2 --max_epochs 300 --valid 5 --seed 2026 --save

python run.py --dataset MKG-W --srank 128 --vrank 128 --trank 128 --gate_dim 128 --alpha_max 0.2 --optimizer AdamW --learning_rate 1e-3 --batch_size 500 --regularizer wN3 --reg 7e-2 --max_epochs 300 --valid 5 --seed 2026 --save

python run.py --dataset DB15K --srank 128 --vrank 128 --trank 128 --gate_dim 128 --alpha_max 0.2 --optimizer AdamW --learning_rate 1e-3 --batch_size 500 --regularizer wN3 --reg 7e-2 --max_epochs 300 --valid 5 --seed 2026 --save