# avgdist-rs

A Rust implementation of a sampling algorithm to estimate the average distance among vertices in graph with a large diameter.



Load it with:
```
git clone git@github.com:massimo-nocentini/avgdist-rs.git
cd avgdist-rs
cargo build --release
```

Then run it with:
```
cargo run --release -- <g> <gt> <th> <e> <tr> <d>
```
where
|key|meaning|
|---|---|
|`g` $\in \mathbb{S}$|the webgraph graph's base name|
|`gt` $\in \mathbb{S}$|the webgraph transposed graph's base name|
|`th` $\in \mathbb{N}$|the number of threads to use|
|`e` $\in \mathbb{N}$|the absolute error $\epsilon$|
|`tr` $\in \lbrace \top, \bot \rbrace$|perform exact computation|
|`d` $\in \lbrace \top, \bot \rbrace$|dummy sampling, just choose vertices at random and do BFS from them|

## Results

|# vertices|edge presence prob|$\epsilon$|dummy|link|
|---|---|---|---|---|
|1k|0.001|0.1|$\top$|[1k-0001p-01e-d.out](https://github.com/massimo-nocentini/avgdist-rs/blob/master/data/erdos-renyi/result/1k-0001p-01e-d.out)|
|1k|0.001|0.1|$\bot$|[1k-0001p-01e.out](https://github.com/massimo-nocentini/avgdist-rs/blob/master/data/erdos-renyi/result/1k-0001p-01e.out)|
|2k|0.001|0.1|$\top$|[2k-0001p-01e-d.out](https://github.com/massimo-nocentini/avgdist-rs/blob/master/data/erdos-renyi/result/2k-0001p-01e-d.out)|
|2k|0.001|0.1|$\bot$|[2k-0001p-01e.out](https://github.com/massimo-nocentini/avgdist-rs/blob/master/data/erdos-renyi/result/2k-0001p-01e.out)|
|5k|0.001|0.1|$\top$|[5k-0001p-01e-d.out](https://github.com/massimo-nocentini/avgdist-rs/blob/master/data/erdos-renyi/result/5k-0001p-01e-d.out)|
|5k|0.001|0.1|$\bot$|[5k-0001p-01e.out](https://github.com/massimo-nocentini/avgdist-rs/blob/master/data/erdos-renyi/result/5k-0001p-01e.out)|


## WebGraph in Rust

The [WebGraph](https://webgraph.di.unimi.it/) framework has been ported to [webgraph-rs](https://github.com/vigna/webgraph-rs) which is written in [Rust](https://www.rust-lang.org/) and we take the opportunity to experiment with it. First install the system with 
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
in order to load, compile and creating the `webgraph` executable, it needs to type the command 
```bash
cargo install webgraph
``` 

### Elias-Fano encodings

First of all we allow random access to successors by creating the Elias-Fano encoding of the graph with the command `webgraph build ef pg`:

```
             Type             Code  Improvement   Weight      Bytes             Bits
       outdegrees            Gamma       0.000%    0.000     0.000                 0
reference_offsets            Unary       0.000%    0.000     0.000                 0
     block_counts            Unary       0.000%    0.000     0.000                 0
           blocks            Unary         NaN%    0.000     0.000                 0
  interval_counts            Unary       5.991%    0.021     4.862M         38897857
  interval_starts    Zeta { k: 5 }      30.884%    0.191    44.900M        359197369
    interval_lens Golomb { b: 13 }       8.671%    0.011     2.683M         21464453
  first_residuals    Zeta { k: 5 }       4.537%    0.316    74.249M        593993096
        residuals            Unary      66.667%    0.461   108.208M        865667642

 Old bit size:      19435000014
 New bit size:      17555779597
   Saved bits:       1879220417
Old byte size:           2.429G
New byte size:           2.194G
  Saved bytes:         234.903M
  Improvement:           9.669%
```
Having the ability to traverse the graph backward, using transposed edges namely, is very important for the implementation of our algorithm; therefore, we create the necessary representations via the command `TMPDIR=/data/bitcoin webgraph transform transpose pg pg-t`. Consequently, we generate the Elias-Fano encoding for the transposed graph too via the command `webgraph build ef pg-t`:

```
             Type             Code  Improvement   Weight      Bytes             Bits
       outdegrees  Golomb { b: 2 }      12.084%    0.445    37.168M        297343088
reference_offsets            Unary       0.000%    0.000     0.000                 0
     block_counts            Unary       0.000%    0.000     0.000                 0
           blocks            Unary         NaN%    0.000     0.000                 0
  interval_counts            Unary       0.074%    0.000    24.270K           194160
  interval_starts    Zeta { k: 4 }      25.414%    0.010   794.136K          6353087
    interval_lens            Gamma       0.000%    0.000     0.000                 0
  first_residuals    Zeta { k: 5 }       3.609%    0.305    25.444M        203555022
        residuals    Zeta { k: 4 }       2.180%    0.241    20.091M        160731919

 Old bit size:      17629244059
 New bit size:      16961066783
   Saved bits:        668177276
Old byte size:           2.204G
New byte size:           2.120G
  Saved bytes:          83.522M
  Improvement:           3.790%
```
