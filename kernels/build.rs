extern crate gcc;

use std::env;

fn main() {
  let out_dir = env::var("OUT_DIR").unwrap();
  let cc = env::var("CC").unwrap_or("gcc".to_owned());
  gcc::Config::new()
    .compiler(&cc)
    .opt_level(3)
    .pic(true)
    .flag("-march=native")
    .flag("--std=gnu99")
    .file("activate.c")
    .file("batchnorm.c")
    .file("conv.c")
    .file("image.c")
    .file("interpolate.c")
    .file("pool.c")
    .compile("libneuralops_kernels.a");
  println!("cargo:rustc-link-search=native={}", out_dir);
}
