export CUDA_VISIBLE_DEVICES=$1 
sl=de
tl=en
dataset='it'
data_dir=${PWD}/outputs/data-bin-join/${dataset}/

epoch=40
save_dir=${PWD}/outputs/${dataset}-${sl}-${tl}-epoch${epoch}/
mkdir -p $save_dir
fairseq-train $data_dir \
              --save-dir $save_dir \
              --arch transformer \
              --source-lang ${sl} --target-lang ${tl} \
              --encoder-layers 6 --decoder-layers 6 \
              --encoder-embed-dim 512 --decoder-embed-dim 512 \
              --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
              --encoder-attention-heads 8 --decoder-attention-heads 8 \
              --encoder-normalize-before --decoder-normalize-before \
              --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
              --weight-decay 0.0001 \
              --label-smoothing 0.2 --criterion label_smoothed_cross_entropy \
              --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
              --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
              --lr 1e-3 --min-lr 1e-9 \
              --max-tokens 2000 \
              --update-freq 8 \
              --max-epoch ${epoch} --save-interval 1 \
              --fp16 \
              --save-interval-updates 5000 1> $save_dir/log 2> $save_dir/err

