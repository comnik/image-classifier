using Mocha

## [MODEL]

conv_layer  = ConvolutionLayer(name="conv1", n_filter=20, kernel=(5,5), bottoms=[:data], tops=[:conv])
pool_layer  = PoolingLayer(name="pool1", kernel=(2,2), stride=(2,2), bottoms=[:conv], tops=[:pool])
conv2_layer = ConvolutionLayer(name="conv2", n_filter=50, kernel=(5,5), bottoms=[:pool], tops=[:conv2])
pool2_layer = PoolingLayer(name="pool2", kernel=(2,2), stride=(2,2), bottoms=[:conv2], tops=[:pool2])
fc1_layer   = InnerProductLayer(name="ip1", output_dim=500, neuron=Neurons.ReLU(), bottoms=[:pool2], tops=[:ip1])
fc2_layer   = InnerProductLayer(name="ip2", output_dim=10, bottoms=[:ip1], tops=[:ip2])

LeNet = [conv_layer, pool_layer, conv2_layer, pool2_layer, fc1_layer, fc2_layer]


## [TRAINING]

backend = CPUBackend()

net = Net("MNIST", backend, [
    HDF5DataLayer(name="train-data", source="../data/mnist/train.txt", batch_size=64, shuffle=true),
    LeNet...,
    SoftmaxLossLayer(name="loss", bottoms=[:ip2,:label])
])

exp_dir = "snapshots"

params = SolverParameters(max_iter   = 10000,
                          regu_coef  = 0.0005,
                          mom_policy = MomPolicy.Fixed(0.9),
                          lr_policy  = LRPolicy.Inv(0.01, 0.0001, 0.75),
                          load_from  = exp_dir)

solver = SGD(params)


## [SNAPSHOTTING]

setup_coffee_lounge(solver, save_into="$exp_dir/statistics.hdf5", every_n_iter=1000)
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)


## [RUN]

solve(solver, net)

destroy(net)
shutdown(backend)
