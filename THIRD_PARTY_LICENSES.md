# Third-Party Licenses / Acknowledgements

The funNystrom++ benchmark (`matlab/benchmark/funnystrompp/`) compares
RandLAPACK's `FunNystromPP` against external reference implementations. **These
are not vendored in this repository.** They are fetched at build time via CMake
`FetchContent`, pinned to a commit each, only when the benchmark is enabled
(`-DRANDLAPACK_BINDINGS_BENCHMARKS=ON`). See
`matlab/benchmark/funnystrompp/CMakeLists.txt`.

## Ethan Epperly — trace-estimation code (XTrace / XNysTrace / resphering)

- Repository: https://github.com/eepperly/Ethan-Epperly-Thesis
- Pinned commit: `2d33ee40577caff18b16fed3454a147d08cbe16e` (branch `main`)
- License: **MIT**.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND ...
```
(Full text in the repository's `LICENSE` file at the pinned commit.)

## David Persson — funNystrom++ reference

- Repository: https://github.com/davpersson/funNystrom
- Pinned commit: `0427bde1e3c238ce53cc184600482481490c2a48` (branch `master`)
- Files used: `Other/nystrom.m`, `Other/funnystrompp.m`, `Other/block_lanczos.m`.
- License: **none stated** at the time of writing (under default copyright, all
  rights reserved). Used here **with attribution**, fetched (not redistributed)
  for benchmark comparison only. **A license is being requested from the
  author**; this acknowledgement is interim.
- Note: `block_lanczos.m` in that repository is Christopher Meyer's
  HutchPlusPlus code, vendored by Persson with provenance noted in its header.

We thank David Persson and Ethan Epperly for making their reference code
available.
