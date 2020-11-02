sl=de
tl=en

sd='it'
td='emea'
data_dir=${PWD}/dataset/${td}-lex-wsub+${sd}
out_dir=${PWD}/outputs/
dest_dir=$out_dir/data-bin-join/${sd}2${td}sub/
mkdir -p $dest_dir

fairseq-preprocess --source-lang ${sl} --target-lang $tl \
	--trainpref $data_dir/${td}-wsub-unsup+${sd}-para-train.bpe \
	--validpref $data_dir/${td}-wsub-unsup+${sd}-para-dev.bpe \
	--testpref $data_dir/${td}-test.bpe \
	--destdir $dest_dir \
	--srcdict $out_dir/data-bin-join/${sd}/dict.${sl}.txt \
	--tgtdict $out_dir/data-bin-join/${sd}/dict.${tl}.txt

