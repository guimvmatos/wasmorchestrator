struct BnParams {
    channels: u32,
    h_in: u32,
    w_in: u32,
    epsilon: f32,
};

@group(0) @binding(0) var<uniform> params: BnParams;
@group(0) @binding(1) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(2) var<storage, read> scale_tensor: array<f32>;
@group(0) @binding(3) var<storage, read> beta_tensor: array<f32>;
@group(0) @binding(4) var<storage, read> mean_tensor: array<f32>;
@group(0) @binding(5) var<storage, read> var_tensor: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_tensor: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x; // posição horizontal
    let y = global_id.y; // posição vertical
    let c = global_id.z; // canal

    if (x >= params.w_in || y >= params.h_in || c >= params.channels) {
        return;
    }

    let c_mean  = mean_tensor[c];
    let c_var   = var_tensor[c];
    let c_scale = scale_tensor[c];
    let c_beta  = beta_tensor[c];

    // Mesma transformação pré-calculada por canal do seu código C:
    // y = (x * inv_std * scale) + (beta - mean * inv_std * scale)
    let inv_std = 1.0 / sqrt(c_var + params.epsilon);
    let a = c_scale * inv_std;
    let b_val = c_beta - (c_mean * a);

    let spatial_size = params.h_in * params.w_in;
    let idx = c * spatial_size + y * params.w_in + x;

    output_tensor[idx] = (input_tensor[idx] * a) + b_val;
}