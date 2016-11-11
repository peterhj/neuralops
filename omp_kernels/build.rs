extern crate gcc;

use std::env;

fn main() {
  let out_dir = env::var("OUT_DIR").unwrap();
  let cc = if cfg!(not(feature = "iomp")) {
    env::var("CC").unwrap_or("gcc".to_owned())
  } else {
    "icc".to_owned()
  };
  gcc::Config::new()
    .compiler(&cc)
    .opt_level(3)
    .pic(true)
    .flag("-std=gnu99")
    .flag("-march=native")
    .flag("-fno-strict-aliasing")
    .flag(if cfg!(not(feature = "iomp")) {
      "-fopenmp"
    } else {
      "-qopenmp"
    })
    .flag("-Isrc")
    .flag("-DNEURALOPS_OMP")
    .file("src/activate.c")
    .compile("libneuralops_omp_kernels.a");
  println!("cargo:rustc-link-search=native={}", out_dir);
}
