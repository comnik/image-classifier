using HDF5

srand(12345678)

dataset = "train.h5"
dataout = "mnist/train.h5"
h       = 28
w       = 28
n_data  = 40000

println("Exporting $n_data digits of size $h x $w")

h5open(dataset, "r") do input
    h5open(dataout, "w") do output
        dset_data = d_create(output, "data", datatype(Float32), dataspace(w, h, 1, n_data))
        dset_label = d_create(output, "label", datatype(Float32), dataspace(1, n_data))

        idx = collect(1:n_data)
        rp = randperm(length(idx))

        img = read(input["data"])
        img = convert(Array{Float32}, img) / 256 # scale into [0,1)
        class = convert(Array{Float32}, read(input["label"]))

        for j = 1:length(idx)
            r_idx = rp[j]
            dset_data[:, :, 1, idx[j]] = img[(r_idx-1)*h*w+1:r_idx*h*w]
            dset_label[1, idx[j]] = class[r_idx]
        end
    end
end
