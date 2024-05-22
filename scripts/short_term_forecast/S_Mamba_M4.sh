model_name=S_Mamba

periods=("Monthly" "Yearly" "Quarterly" "Weekly" "Daily" "Hourly")
for period in "${periods[@]}";
do

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --pre_batch 32 \
  --pre_out 128 \
  --root_path ./dataset/m4 \
  --seasonal_patterns $period \
  --model_id m4_$period \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --enc_in 1 \
  --expand 2 \
  --d_state 16 \
  --d_conv 4 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'  

done