python main.py --model=resnet-18 --epoch=75 --steps=49,50,63 --batch_size=128 --sched=warmup \
    --min_lr=0.001 --max_lr=0.03 --prune=lottery-simp --pruning_count=20 --strategy=linear --thread=1 | tee resnet18-simp0.03.log

python main.py --model=resnet-18 --epoch=75 --steps=49,50 --batch_size=128 --sched=warmup \
    --min_lr=0.001 --max_lr=0.03 --prune=lottery-simp --pruning_count=20 --strategy=cos --thread=1 | tee resnet18-simpcos0.03.log

python main.py --model=resnet-18 --epoch=75 --steps=49,50,63 --batch_size=128 --sched=warmup \
    --min_lr=0.001 --max_lr=0.03 --prune=lottery --pruning_count=20 --strategy=linear --thread=1 | tee resnet18-paper0.03.log

python main.py --model=resnet-18 --epoch=75 --steps=49,50,63 --batch_size=128 --sched=warmup \
    --min_lr=0.001 --max_lr=0.1 --prune=lottery-simp --pruning_count=20 --strategy=linear --thread=1 | tee resnet18-simp0.1.log

python main.py --model=resnet-18 --epoch=75 --steps=49,50,63 --batch_size=128 --sched=warmup \
    --min_lr=0.001 --max_lr=0.1 --prune=lottery --pruning_count=20 --strategy=linear --thread=1 | tee resnet18-paper0.1.log

python main.py --model=efficientnet-b0 --epoch=75 --steps=49,50,63 --batch_size=128 --sched=warmup \
    --min_lr=0.001 --max_lr=0.1 --prune=lottery-simp --pruning_count=20 --strategy=linear --thread=1 | tee effb0-simp0.1.log

python main.py --model=efficientnet-b0 --epoch=75 --steps=49,50,63 --batch_size=128 --sched=warmup \
    --min_lr=0.001 --max_lr=0.1 --prune=lottery --pruning_count=20 --strategy=linear --thread=1 | tee effb0-paper0.1.log

python main.py --model=resnet-50 --epoch=75 --steps=49,50,63 --batch_size=128 --sched=warmup \
    --min_lr=0.001 --max_lr=0.1 --prune=lottery-simp --pruning_count=20 --strategy=linear --thread=1 | tee resnet50-simp0.1.log

python main.py --model=resnet-50 --epoch=75 --steps=49,50,63 --batch_size=128 --sched=warmup \
    --min_lr=0.001 --max_lr=0.1 --prune=lottery --pruning_count=20 --strategy=linear --thread=1 | tee resnet50-paper0.1.log

python main.py --model=resnet-50 --epoch=75 --steps=10 --batch_size=128 --sched=onecycle \
    --min_lr=0.05 --max_lr=0.8 --prune=lottery-simp --pruning_count=20 --strategy=cos --thread=1 | tee resnet50-cycle.log

python main.py --model=efficientnet-b5 --epoch=75 --steps=10 --batch_size=128 --sched=onecycle \
    --min_lr=0.05 --max_lr=0.8 --prune=lottery-simp --pruning_count=20 --strategy=cos --thread=1 | tee effb5-cycle.log

python main.py --model=efficientnet-b5 --epoch=75 --steps=49,50,63 --batch_size=128 --sched=warmup \
    --min_lr=0.05 --max_lr=0.8 --prune=lottery-simp --pruning_count=20 --strategy=cos --thread=1 | tee effb5-paper.log