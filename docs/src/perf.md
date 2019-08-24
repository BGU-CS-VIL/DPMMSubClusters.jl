# Performance Tips

This package support multiple processes, over multiple machines, and will use all the available worker processes.
Note that when running without any worker processes it will use the master process.

For optimal performance, run with `BLAS.set_num_threads(1)`.

The performance increase is not linear with the processes, and on small data sets of lower dimensions adding more processes might even reduce performance.

Optimization contributions are very welcomed!.
