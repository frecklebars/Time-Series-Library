model_name=S_Mamba
pretrain_strategy=ContrastiveBasic

for pred_len in 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_pretraining 1 \
  --pretrain_strategy $pretrain_strategy \
  --pre_batch 32 \
  --pre_accumulation_steps 16 \
  --pre_out 128 \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00007

done