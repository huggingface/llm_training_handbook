# How to Maximize Training Throughput

The faster you can make your model to train the sooner the model will finish training, which is important not only to being first to publish something, but also potentially saving a lot of money.

In general maximizing throughput is all about running many experiments and measuring the outcome and chosing the one that is superior.

In certain situations your modeling team may ask you to choose some hyper parameters that will be detrimental to throughput but overall beneficial for the overall model's success.

## Crucial reproducibility requirements

The most important requirements for a series of successful experiments is to be able to reproduce the experiment environment again and again while changing only one or a few setup variables.

Therefore when you try to figure out whether some change will improve performance or make it worse, you must figure out how to keep things stable.

For example, you need to find a way to prevent the network usage from fluctuations. When we were doing performance optimizations for [108B pre-BLOOM experiments](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide) it was close to impossible to perform, since we were on a shared internode network and the exact same setup would yield different throughput depending on how many other users used the network. It was not working. During BLOOM-176B we were given a dedicated SLURM partition with an isolated network where the only traffic was ours. Doing the performance optimization in such environment was just perfect.

## Network throughput

It's critical to understand your particular model size and framework requirements with regard to network bandwidth, throughput and latency. If you underpay for network you will end up having idle gpus and thus you wasted money and time. If you overpay for very fast network, but your gpus are slow, then again you wasted money and time.

If your network is very slow, your training is likely to be network-bound and many improvements in the training setup will not help with the improving performance.

Here is a simple all-reduce benchmark that you can use to quickly measure the throughput of your internode network:

[all_reduce_bench.py](./all_reduce_bench.py)

Usually benchmarking at least 4 nodes is recommended, but, of course, if you already have access to all the nodes you will be using during the training, benchmark using all of the nodes.

To run it on 4 nodes

```
python -m torch.distributed.run --nproc_per_node=4 all_reduce_bench.py
```

You may get results anywhere between 5Gbps and 1600Gbps (as of this writing). The minimal speed to prevent being network bound will depend on your particular training framework, but typically you'd want at least 400Gbps or higher. Though we trained BLOOM on 50Gbps.

Frameworks that shard weights and optim stages like [Deepspeed](https://github.com/microsoft/DeepSpeed) w/ ZeRO Stage-3 do a lot more traffic than frameworks like [Megatron-Deepspeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed) which do tensor and pipeline parallelism in addition to data parallelism. The latter ones only send activations across and thus don't need as much bandwidth. But they are much more complicated to set up and run.

Of course, an efficient framework will overlap communications and compute, so that while one stage is fetching data, the other stage in parallel runs computations. So as long as the communication overhead is smaller than compute the network requirements are satisfied and don't have to be super fantastic.

To get reasonable GPU throughput when training at scale (64+GPUs) with DeepSpeed ZeRO Stage 3:

1. 100Gbps is not enough
2. 200-400 Gbps is ok
3. 800-1000 Gbps is ideal

[full details](https://github.com/microsoft/DeepSpeed/issues/2928#issuecomment-1463041491)


## TFLOPs as a performance metric

Before you start optimizing the performance of your training setup you need a metric that you can use to see whether the throughput is improving or not. You can measure seconds per iteration, or iterations per second, or some other such timing, but there is a more useful metric that measures TFLOPs.

footnote: TFLOPs: Trillion FLOPs per second - [FLOPS](https://en.wikipedia.org/wiki/FLOPS)

Measuring TFLOPs is superior because without it you don't know whether you are close to the best performance that can be achieved or not. This measurement gives you an indication of how far you're from the peak performance reported by the hardware manufacturer.

In this section I will use BLOOM's training for the examplification. We use 80GB A100 NVIDIA GPUs and we trained in mixed bf16 regime. So let's look at the [A100 spec](https://www.nvidia.com/en-us/data-center/a100/) which tells us:

```
BFLOAT16 Tensor Core 	312 TFLOPS
```

Therefore we now know that if we were to only run `matmul` on huge bf16 matrices without copying to and from the device we should get around 312 TFLOPs max.

Practically though, due to disk IO, communications and copying data from gpu memory to gpu computing unit overhead and because we can't do everything in bf16 and at times we have to do math in fp32 (or tf32) we can really expect about half of that. So 155 TFLOPs should be an amazing sustainable throughput for a complex hundreds of GPUs training setup.

When we first started tuning things up we were at <100 TFLOPs and a few weeks later when we launched the training we managed to get 150 TFLOPs.

The important thing to notice here is that we knew that we can't push it further by much and we knew that there was no more point to try and optimize it even more.

So a general rule of thumb - if your training set up gets about 1/2 of advertised peak performance you're doing great. Don't let it stop you though from beating this suggestion and getting even more efficient.

When calculating TFLOPs it's important to remember that the math is different if [Checkpoint activations](#checkpoint-activations) are enabled, since when it's activated more compute is used and it needs to be taken into an account.

for transformer models the following is an estimation formula which slightly under-reports the real TFLOPs:

TFLOPs: `model_size_in_B * 4 * 2 * seqlen * global_batch_size / (time_in_sec_per_interation * total_gpus * 1e3)`

The factor of 4 is when used with activation check-pointing, otherwise it will be 3, but for 100B+ model, activation check-pointing will always be on.

So the `3*2` is often called "model FLOPs" and `4*2` - "hardware FLOPs".

```
perl -le '$ng=64; $ms=52; $gbs=1024; $sp=127; $seqlen=2048; print $ms*4*2*$seqlen*$gbs / ( $sp * $ng * 1e3)'
```
(ng = total gpus, ms = model size in B, gbs = global batch size, sp = throughput in seconds)

same with bash env vars and broken down GBS into mbs*dp*gas (gas=pp_chunks):
```
echo "($MSIZE*4*2*SEQLEN*$MICRO_BATCH_SIZE*$DP_SIZE*$GAS)/($THROUGHPUT*$NNODES*4*1000)" | bc -l
```

The exact formula is in Equation 3 of Section 5.1 of the [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) paper. You can see the code [here](https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/251).

footnote: For Inference only it'd be: `24Bsh^2 + 4𝐵s^2h` floating point operations per layer




## Checkpoint activations

Enabling checkpoint activations allows one to trade speed for memory. When this feature is activated instead of remembering the outputs of, say, transformer blocks until the backward pass is done, these outputs are dropped. This frees up huge amounts of GPU memory. But, of course, a backward pass is not possible without having the outputs of forward pass, and thus they have to be recalculated.

This, of course, can vary from model to model, but typically one pays with about 20-25% decrease in throughput, but since a huge amount of gpu memory is liberated, one can now increase the batch size per gpu and thus overall improve the effective throughput of the system.



## Gradient accumulation

Depending on a situation using a large gradient accumulation can increase the throughput, even though it's only the optimizer `step` that's skipped except at the boundary of the gradient accumulation, it can be quite a significant saving. e.g. in this particular small setup I clocked 20-30% speed up:

- [A100](https://github.com/huggingface/transformers/issues/15026#issuecomment-1004592231)
- [RTX-3090](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004392537)


When using Pipeline parallelism a very large Gradient Accumulation is a must to keep the [pipeline's bubble to the minimum]( https://huggingface.co/docs/transformers/main/perf_train_gpu_many#naive-model-parallelism-vertical-and-pipeline-parallelism).




## Vector and matrix size divisibility


### Tile and wave quantization

XXX


### Number/size of Attention heads

XXX
