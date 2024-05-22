
model_name=S_Mamba

for pred_len in 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 8 \
  --expand 2 \
  --d_state 16 \
  --d_conv 4 \
  --c_out 8 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \

done