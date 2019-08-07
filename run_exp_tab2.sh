gpuid=0

dataset='roxford5k'
# dataset='rparis6k'
# dataset='copydays'
# dataset='holidays'

dir_cache='data/'
log='data/log_tab2_'$dataset'.txt'

# attack params
scales='[300,350,400,450,500,550,600,650,700,750,800,850,900,950,1024]'
carrier='flower'
lam=0.0


# attack params
arch='alexnet'
sigmablur=0.3
iter=100
mode='hist'
modellist=$arch"-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid
# test params
testscale=1024
testpool='gem'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log


# attack params
arch='resnet18'
sigmablur=0.3
iter=100
mode='global'
pool='gem'
modellist=$arch"-"$pool
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid
# test params
testscale=1024
testpool='gem'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
# test params
testscale=768
testpool='gem'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
# test params
testscale=512
testpool='gem'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log



# attack params
arch='resnet18'
sigmablur=0.0
iter=100
mode='hist'
modellist=$arch"-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid
# test params
testscale=1024
testpool='gem'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
# test params
testscale=768
testpool='gem'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
# test params
testscale=512
testpool='gem'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log


# attack params
arch='resnet18'
sigmablur=0.3
iter=100
mode='hist'
modellist=$arch"-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid
# test params
testscale=1024
testpool='gem'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
# test params
testscale=768
testpool='gem'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
# test params
testscale=512
testpool='gem'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log





# attack params
arch='resnet18'
sigmablur=0.3
iter=100
mode='global'
pool='gem-mac-spoc'
modellist=$arch"-"$pool
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid
# test params
testscale=1024
testpool='crow'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log


# attack params
arch='resnet18'
sigmablur=0.3
iter=100
mode='hist'
modellist=$arch"-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid
# test params
testscale=1024
testpool='crow'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log


# attack params
arch='resnet18'
sigmablur=0.3
iter=1000
mode='tensor'
modellist=$arch"-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid
# test params
testscale=1024
testpool='crow'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log


#attack params
sigmablur=0.3
iter=100
mode='hist'
modellist="alexnet-"$mode"+resnet18-"$mode
#attack
python3 attack_queries.py --dataset=$dataset --carrier=$carrier --mode=$mode --modellist=$modellist --scales=$scales --iters=$iter --lam=$lam --sigma-blur=$sigmablur --gpu-id=$gpuid
# test params
testscale=1024
testpool='gem'
arch='alexnet'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
# test params
testscale=1024
testpool='crow'
arch='resnet18'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
# test params
testscale=1024
testpool='gem'
arch='vgg16'
# evaluate retrieval performance
python3 eval_retrieval.py --gpu-id $gpuid --network-offtheshelf $arch"-"$testpool --dataset $dataset --image-size 1024 --image-resize $testscale --dir-attack "data/"$dataset"_"$modellist"+"$scales"+iter"$iter"+lr0.01+lam"$lam"+sigmablur"$sigmablur"_"$carrier"/" --ext-attack '.png' --dir-cache $dir_cache --log=$log
