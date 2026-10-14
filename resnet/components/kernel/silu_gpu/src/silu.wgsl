@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input_tensor)) {
        return;
    }
    let x = input_tensor[idx];
    let sigmoid_x = 1.0 / (1.0 + exp(-x));
    output_tensor[idx] = x * sigmoid_x;
}