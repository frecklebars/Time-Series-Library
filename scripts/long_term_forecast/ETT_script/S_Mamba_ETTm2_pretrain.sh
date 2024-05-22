# export CUDA_VISIBLE_DEVICES=2

model_name=S_Mamba
for pred_len in 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_pretraining 1 \
  --pre_batch 32 \
  --pre_accumulation_steps 4 \
  --pre_out 128 \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_state 2\
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00005

done

for pred_len in 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_pretraining 1 \
  --pre_batch 32 \
  --pre_accumulation_steps 8 \
  --pre_out 128 \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_state 2\
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00005

done
