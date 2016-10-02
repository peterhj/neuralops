extern crate gcc;

fn main() {
  gcc::Config::new()
    .compiler("gcc-4.9")
    .opt_level(3)
    .pic(true)
    .flag("-march=native")
    .flag("--std=gnu99")
    .file("extkernels/batchnorm.c")
    .file("extkernels/conv.c")
    .file("extkernels/pool.c")
    .compile("libneuralops_extkernels.a");

  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64");
}
