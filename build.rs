extern crate gcc;

fn main() {
  gcc::Config::new()
    .compiler("gcc-4.9")
    .opt_level(3)
    .pic(true)
    //.flag("-march=native")
    .flag("--std=gnu99")
    .file("extkernels/activate.c")
    .file("extkernels/batchnorm.c")
    .file("extkernels/conv.c")
    .file("extkernels/image.c")
    .file("extkernels/interpolate.c")
    .file("extkernels/pool.c")
    .compile("libneuralops_extkernels.a");

  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64");
  println!("cargo:rustc-link-search=native={}", "/opt/intel/mkl/lib/intel64");
}
