@group(0) @binding(0) var<storage, read> a_tensor: array<f32>;
@group(0) @binding(1) var<storage, read> b_tensor: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_tensor: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len_a = arrayLength(&a_tensor);
    if (idx >= len_a) {
        return;
    }
    let len_b = arrayLength(&b_tensor);
    // Mesmo broadcast por módulo do kernel CPU: elementwise se len_b==len_a,
    // escalar se len_b==1. Sem broadcast multi-dimensional real.
    output_tensor[idx] = a_tensor[idx] * b_tensor[idx % len_b];
}
