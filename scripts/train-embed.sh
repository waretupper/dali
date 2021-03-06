repo=$PWD
data_dir=$repo/dataset/

# combine unaligned monolingual data in both languages
echo "combine monolingual data"
cat ${data_dir}/*-train.tc.clean.en.mono > ${data_dir}/all-train.tc.clean.mono.en
cat ${data_dir}/*-train.tc.clean.de.mono > ${data_dir}/all-train.tc.clean.mono.de
prefix=all-train.tc.clean.mono

# train embeddings on all the unaligned monolingual data in two languages
fasttext="$repo/fastText/build/fasttext"
out_dir="$repo/embed/"
mkdir -p $out_dir 
for lang in de en; do  
    echo "$fasttext skipgram -input ${data_dir}/${prefix}.${lang} -output ${out_dir}/${prefix}.${lang} -ws 10 -dim 512 -neg 10 -t 0.00001 -epoch 10"
    $fasttext skipgram -input ${data_dir}/${prefix}.${lang} -output ${out_dir}/${prefix}.${lang} -ws 10 -dim 512 -neg 10 -t 0.00001 -epoch 10 
done