extern crate gcc;

use std::env;

fn main() {
  let out_dir = env::var("OUT_DIR").unwrap();
  gcc::Config::new()
    .compiler("icc")
    .opt_level(3)
    .pic(true)
    .flag("-std=gnu99")
    .flag("-march=native")
    .flag("-qopenmp")
    .compile("libneuralops_iomp_kernels.a");
  println!("cargo:rustc-link-search=native={}", out_dir);
}
