struct GemmParams {
    in_features: u32,
    out_features: u32,
};

@group(0) @binding(0) var<uniform> params: GemmParams;
@group(0) @binding(1) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(2) var<storage, read> weights_tensor: array<f32>;
@group(0) @binding(3) var<storage, read> bias_tensor: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_tensor: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x; // índice do neurônio de saída

    if (i >= params.out_features) {
        return;
    }

    var sum: f32 = bias_tensor[i];
    let w_offset = i * params.in_features;

    for (var j: u32 = 0u; j < params.in_features; j = j + 1u) {
        sum = sum + input_tensor[j] * weights_tensor[w_offset + j];
    }

    output_tensor[i] = sum;
}