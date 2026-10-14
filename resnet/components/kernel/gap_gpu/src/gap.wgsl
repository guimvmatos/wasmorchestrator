struct Params {
    channels: u32,
    spatial_size: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_tensor: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let c = global_id.x;

    if (c >= params.channels) {
        return;
    }

    var sum: f32 = 0.0;
    let base = c * params.spatial_size;

    for (var i: u32 = 0u; i < params.spatial_size; i = i + 1u) {
        sum = sum + input_tensor[base + i];
    }

    output_tensor[c] = sum / f32(params.spatial_size);
}