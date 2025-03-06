#[unsafe(no_mangle)]
pub extern "C" fn matrix_mul(a: *mut f32, b: *mut f32, c: *mut f32, n: usize) {
    let a_slice = unsafe { std::slice::from_raw_parts(a, n * n) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, n * n) };
    let c_slice = unsafe { std::slice::from_raw_parts_mut(c, n * n) };

    for i in 0..n {
        for j in 0..n {
            c_slice[i * n + j] = 0.0;
            for k in 0..n {
                c_slice[i * n + j] += a_slice[i * n + k] * b_slice[k * n + j];
            }
        }
    }
}

fn main() {}