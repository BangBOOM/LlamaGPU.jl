using Mmap
using BFloat16s
using ProgressMeter

function convert(src, tgt, dim)
    f_src = open(src, "r")
    f_tgt = open(tgt, "w")
    t = Matrix{Float32}(undef, dim, 1)
    round, left = divrem(stat(f_src).size // 4, dim)
    pb = Progress(round)
    for i in 1:round
        read!(f_src, t)
        write(f_tgt, BFloat16.(t))
        next!(pb)
    end
    t_left = Matrix{Float32}(undef, Int(left), 1)
    read!(f_src, t_left)
    write(f_tgt, BFloat16.(t_left))
    close(f_src)
    close(f_tgt)
end

convert(ARGS[1], ARGS[2], parse(Int, ARGS[3]))