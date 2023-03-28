## First experiment
python ensemble_adver_train_mnist.py --model modelA --type=0 --namestr="" --train_set train

python ensemble_adver_train_mnist.py --model modelB --type=1 --namestr="" --train_set train

python ensemble_adver_train_mnist.py --model modelC --type=2 --namestr="" --train_set test

python ensemble_adver_train_mnist.py --model modelD --type=3 --namestr="" --train_set train

python no_box_attack.py --dataset mnist --namestr="C Mnist eps=0.3 Extragradient PGD-Critic=True Lambda=10 Training-set=test" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 1024 --test_batch_size 64 --attack_epochs 100 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model CondGen --command train --source_arch ens_adv --model_name modelC --type 2 --eval_freq 5 --transfer --lambda_on_clean 10 --pgd_on_critic --save_model "modelC-pgd-critic-test" --dir_test_models ../ --adv_models modelC --train_set test --wandb

python no_box_attack.py --dataset mnist --namestr="C Mnist eps=0.3 Extragradient PGD-Critic=True Lambda=10 Training-set=test" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 1024 --test_batch_size 64 --attack_epochs 100 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model CondGen --command eval --source_arch ens_adv --model_name modelC --type 2 --transfer --lambda_on_clean 10 --pgd_on_critic --save_model "modelC-pgd-critic-test" --dir_test_models ../ --adv_models modelA modelB modelC modelD

python no_box_attack.py --dataset mnist --namestr="C Mnist eps=0.3 Extragradient PGD-Critic=True Lambda=10 Training-set=test" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 1024 --test_batch_size 64 --attack_epochs 100 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model CondGen --command eval --source_arch ens_adv --model_name modelC --type 2 --transfer --lambda_on_clean 10 --pgd_on_critic --save_model "modelC-pgd-critic-test" --dir_test_models ../ --adv_models modelA modelB modelC modelD modelA_adv modelB_adv modelC_adv modelD_adv


##  Second experiment
python ensemble_adver_train_mnist.py --model modelA_adv --type=0 --namestr="" --train_set train --train_adv --adv_models --epsilon 0.3 --seed 42

python ensemble_adver_train_mnist.py --model modelB_adv --type=1 --namestr="" --train_set train --train_adv --adv_models --epsilon 0.3 --seed 128

python ensemble_adver_train_mnist.py --model modelC_adv --type=2 --namestr="" --train_set test --train_adv --adv_models --epsilon 0.3 --seed 256

python ensemble_adver_train_mnist.py --model modelD_adv --type=3 --namestr="" --train_set train --train_adv --adv_models --epsilon 0.3 --seed 12


python no_box_attack.py --dataset mnist --namestr="C Mnist eps=0.3 Extragradient PGD-Critic=True Lambda=10 Training-set=test with adv" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 1024 --test_batch_size 64 --attack_epochs 150 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model CondGen --command train --source_arch ens_adv --model_name modelC --type 2 --eval_freq 10 --transfer --lambda_on_clean 10 --pgd_on_critic --save_model "modelC-pgd-critic-test-adv" --dir_test_models ../ --adv_models modelC_adv --train_set test --wandb

python no_box_attack.py --dataset mnist --namestr="C Mnist eps=0.3 Extragradient PGD-Critic=True Lambda=10 Training-set=test with adv" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 1024 --test_batch_size 64 --attack_epochs 100 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model CondGen --command eval --source_arch ens_adv --model_name modelC --type 2 --transfer --lambda_on_clean 10 --pgd_on_critic --save_model "modelC-pgd-critic-test-adv" --dir_test_models ../ --adv_models modelA modelB modelC modelD modelA_adv modelB_adv modelC_adv modelD_adv

 
##  Third experiment (With implemented architecture with the batch normalization)
python ensemble_adver_train_mnist.py --model modelA --type=0 --namestr="" --train_set train --seed 76

python ensemble_adver_train_mnist.py --model modelB --type=1 --namestr="" --train_set train --seed 59

python ensemble_adver_train_mnist.py --model modelC --type=2 --namestr="" --train_set train --seed 589 (eval on modelC trained with the train set)

python ensemble_adver_train_mnist.py --model modelD --type=3 --namestr="" --train_set train --seed 512

python no_box_attack.py --dataset mnist --namestr="C Mnist eps=0.3 Extragradient PGD-Critic=True Lambda=10 batchnorm" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 1024 --test_batch_size 64 --attack_epochs 100 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model CondGen --command eval --source_arch ens_adv --model_name modelC --type 2 --transfer --lambda_on_clean 10 --pgd_on_critic --save_model "modelC-pgd-critic-test" --dir_test_models ../ --adv_models modelA modelB modelC modelD

python no_box_attack.py --dataset mnist --namestr="C Mnist eps=0.3 Extragradient PGD-Critic=True Lambda=10 batchnorm with adv" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 1024 --test_batch_size 64 --attack_epochs 100 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model CondGen --command eval --source_arch ens_adv --model_name modelC --type 2 --transfer --lambda_on_clean 10 --pgd_on_critic --save_model "modelC-pgd-critic-test-adv" --dir_test_models ../ --adv_models modelA modelB modelC modelD


##  Fourth experiment (Convergence of extragradient)
python no_box_attack.py --dataset mnist --namestr="C Mnist eps=0.3 PGD-Critic=True Lambda=10 Training-set=test without extragradient" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 1024 --test_batch_size 64 --attack_epochs 100 --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model CondGen --command train --source_arch ens_adv --model_name modelC --type 2 --eval_freq 5 --transfer --lambda_on_clean 10 --pgd_on_critic --save_model "modelC-pgd-critic-test-without-extragradient" --dir_test_models ../ --adv_models modelC --train_set test --wandb


##  Fifth experiment (Adversarial training)
python no_box_attack.py --dataset mnist --namestr="C Mnist eps=0.3 PGD-Critic=True Lambda=10 Training-set=test GenSamples" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 1024 --test_batch_size 64 --attack_epochs 100 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model CondGen --command perturb --source_arch ens_adv --model_name modelC --type 2 --eval_freq 5 --transfer --lambda_on_clean 10 --pgd_on_critic --save_model "modelC-pgd-critic-test" --dir_test_models ../ --adv_models modelC

python no_box_attack.py --dataset mnist --namestr="C Mnist eps=0.3 PGD-Critic=True Lambda=10 Training-set=test GenSamples" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 1024 --test_batch_size 64 --attack_epochs 100 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model CondGen --command perturb --source_arch ens_adv --model_name modelC --type 2 --eval_freq 5 --transfer --lambda_on_clean 10 --pgd_on_critic --save_model "modelC-pgd-critic-test-adv" --dir_test_models ../ --adv_models modelC

python ensemble_adver_train_mnist.py --model modelA --type=0 --namestr="" --self_loader --seed 11

python ensemble_adver_train_mnist.py --model modelB --type=1 --namestr="" --self_loader --seed 176

python ensemble_adver_train_mnist.py --model modelC --type=2 --namestr="" --self_loader --seed 78

python ensemble_adver_train_mnist.py --model modelA_adv --type=0 --namestr="" --train_set test --train_adv --adv_models --epsilon 0.3 --seed 42

python ensemble_adver_train_mnist.py --model modelB_adv --type=1 --namestr="" --train_set test --train_adv --adv_models --epsilon 0.3 --seed 128

python ensemble_adver_train_mnist.py --model modelC_adv --type=2 --namestr="" --train_set test --train_adv --adv_models --epsilon 0.3 --seed 256

python no_box_attack.py --dataset mnist --namestr="C Mnist eps=0.3 PGD-Critic=True Lambda=10 Training-set=test Eval adv train" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 1024 --test_batch_size 64 --attack_epochs 100 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model CondGen --command eval --source_arch ens_adv --model_name modelC --type 2 --eval_freq 5 --transfer --lambda_on_clean 10 --pgd_on_critic --save_model "modelC-pgd-critic-test" --dir_test_models ../ --adv_models modelA modelB modelC modelA_adv modelB_adv modelC_adv

python no_box_attack.py --dataset mnist --namestr="C Mnist eps=0.3 PGD-Critic=True Lambda=10 Training-set=test Eval adv train" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 1024 --test_batch_size 64 --attack_epochs 100 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model CondGen --command eval --source_arch ens_adv --model_name modelC --type 2 --eval_freq 5 --transfer --lambda_on_clean 10 --pgd_on_critic --save_model "modelC-pgd-critic-test-adv" --dir_test_models ../ --adv_models modelA modelB modelC modelA_adv modelB_adv modelC_adv
