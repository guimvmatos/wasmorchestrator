struct Params {
    channels: u32,
    h_in: u32,
    w_in: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    padding: u32,
    out_h: u32,
    out_w: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_tensor: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x; // posição horizontal de saída
    let y = global_id.y; // posição vertical de saída
    let c = global_id.z; // canal

    if (x >= params.out_w || y >= params.out_h || c >= params.channels) {
        return;
    }

    // Sentinela de "menos infinito" prático (posições de padding nunca vencem).
    var max_val: f32 = -3.4028235e38;

    let base_y = i32(y * params.stride) - i32(params.padding);
    let base_x = i32(x * params.stride) - i32(params.padding);

    for (var ky: u32 = 0u; ky < params.kh; ky = ky + 1u) {
        let iy = base_y + i32(ky);
        if (iy < 0 || iy >= i32(params.h_in)) {
            continue;
        }
        for (var kx: u32 = 0u; kx < params.kw; kx = kx + 1u) {
            let ix = base_x + i32(kx);
            if (ix < 0 || ix >= i32(params.w_in)) {
                continue;
            }
            let idx = c * params.h_in * params.w_in + u32(iy) * params.w_in + u32(ix);
            max_val = max(max_val, input_tensor[idx]);
        }
    }

    let out_idx = c * params.out_h * params.out_w + y * params.out_w + x;
    output_tensor[out_idx] = max_val;
}