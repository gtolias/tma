gpuid=0
dataset='rparis6k'

dir_cache='data/'
log='log_fig5_'$dataset'.txt'

# attack params
scales_s0='[1024]'
scales_s1='[300,400,500,600,700,800,900,1024]'
scales_s2='[300,350,400,450,500,550,600,650,700,750,800,850,900,950,1024]'
scales_s3='[262,289,319,351,387,427,470,518,571,630,694,765,843,929,1024]'
carrier='flower'
lam=0.0


# attack params
scales=$scales_s0
arch='alexnet'
sigmablur=0.0
iter=100
mode='hist'
modellist=$arch"-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid

# test params
testpool='gem'
for testscale in $(seq 250 10 1020) 1024; 
do
	echo $testscale; 
	# evaluate retrieval performance
	python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
done 


# attack params
scales=$scales_s0
arch='alexnet'
sigmablur=0.3
iter=100
mode='hist'
modellist=$arch"-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid

# test params
testpool='gem'
for testscale in $(seq 250 10 1020) 1024; 
do
	echo $testscale; 
	# evaluate retrieval performance
	python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
done 


# attack params
scales=$scales_s1
arch='alexnet'
sigmablur=0.0
iter=100
mode='hist'
modellist=$arch"-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid

# test params
testpool='gem'
for testscale in $(seq 250 10 1020) 1024; 
do
	echo $testscale; 
	# evaluate retrieval performance
	python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
done 


# attack params
scales=$scales_s2
arch='alexnet'
sigmablur=0.0
iter=100
mode='hist'
modellist=$arch"-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid

# test params
testpool='gem'
for testscale in $(seq 250 10 1020) 1024; 
do
	echo $testscale; 
	# evaluate retrieval performance
	python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
done 


# attack params
scales=$scales_s1
arch='alexnet'
sigmablur=0.3
iter=100
mode='hist'
modellist=$arch"-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid

# test params
testpool='gem'
for testscale in $(seq 250 10 1020) 1024; 
do
	echo $testscale; 
	# evaluate retrieval performance
	python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
done 


# attack params
scales=$scales_s2
arch='alexnet'
sigmablur=0.3
iter=100
mode='hist'
modellist=$arch"-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid

# test params
testpool='gem'
for testscale in $(seq 250 10 1020) 1024; 
do
	echo $testscale; 
	# evaluate retrieval performance
	python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
done 


# attack params
scales=$scales_s3
arch='alexnet'
sigmablur=0.3
iter=100
mode='hist'
modellist=$arch"-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid

# test params
testpool='gem'
for testscale in $(seq 250 10 1020) 1024; 
do
	echo $testscale; 
	# evaluate retrieval performance
	python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
done 
