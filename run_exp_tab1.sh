gpuid=0

dir_cache='data2/'
log='data/log_tab1.txt'

# attack params
dataset='rparis6k'
carrier='flower'
arch='alexnet'
sigmablur=0.0
scales='[1024]'
lam=0.0

# test params
testscale=1024


# attack params
iter=100
mode='global'
pool='gem'
modellist=$arch"-"$pool
# run attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid
# evaluate retrieval performance
testpool='gem'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='mac'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='spoc'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='rmac'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='crow'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log


# attack params
iter=100
mode='global'
pool='gem-mac-spoc'
modellist=$arch"-"$pool
# run attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$arch"-"$pool --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid
# evaluate retrieval performance
testpool='gem'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='mac'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='spoc'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='rmac'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='crow'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log


# attack params
iter=100
mode='hist'
pool=$mode
modellist=$arch"-"$pool
# run attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$arch"-"$pool --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid
# evaluate retrieval performance
testpool='gem'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='mac'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='spoc'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='rmac'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='crow'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log


# attack params
iter=1000
mode='tensor'
pool=$mode
modellist=$arch"-"$pool
# run attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$arch"-"$pool --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid
# evaluate retrieval performance
testpool='gem'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='mac'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='spoc'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='rmac'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
testpool='crow'
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
