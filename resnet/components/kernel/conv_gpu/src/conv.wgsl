struct ConvParams {
    c_out: u32,
    c_in: u32,
    h_in: u32,
    w_in: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    padding: u32,
    out_h: u32,
    out_w: u32,
};

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(2) var<storage, read> weights_tensor: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_tensor: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x; // pixel horizontal de saída (j)
    let out_y = global_id.y; // pixel vertical de saída (i)
    let co    = global_id.z; // canal de saída (co)

    if (out_x >= params.out_w || out_y >= params.out_h || co >= params.c_out) {
        return;
    }

    var sum: f32 = 0.0;
    
    let in_base_h = i32(out_y * params.stride) - i32(params.padding);
    let in_base_w = i32(out_x * params.stride) - i32(params.padding);

    for (var ci: u32 = 0u; ci < params.c_in; ci = ci + 1u) {
        for (var ki: u32 = 0u; ki < params.kh; ki = ki + 1u) {
            for (var kj: u32 = 0u; kj < params.kw; kj = kj + 1u) {
                let in_h = in_base_h + i32(ki);
                let in_w = in_base_w + i32(kj);

                if (in_h >= 0 && in_h < i32(params.h_in) && in_w >= 0 && in_w < i32(params.w_in)) {
                    let input_idx = ci * (params.h_in * params.w_in)
                                  + u32(in_h) * params.w_in
                                  + u32(in_w);

                    let kernel_idx = co * (params.c_in * params.kh * params.kw)
                                   + ci * (params.kh * params.kw)
                                   + ki * params.kw
                                   + kj;

                    sum = sum + input_tensor[input_idx] * weights_tensor[kernel_idx];
                }
            }
        }
    }

    let out_idx = co * (params.out_h * params.out_w)
                + out_y * params.out_w
                + out_x;

    output_tensor[out_idx] = sum;
}
