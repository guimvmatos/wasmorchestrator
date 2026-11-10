// Conv + BatchNormalization (+ Relu opcional) fundidos.
// Mesma convolução agrupada do conv.wgsl; o BN (e o bias do Conv) chegam
// já reduzidos a dois vetores por canal de saída, calculados no host:
//   bn_a[co] = scale / sqrt(var + eps)
//   bn_b[co] = beta - mean * bn_a[co] + bias[co] * bn_a[co]
//   out = sum * bn_a[co] + bn_b[co]   (e max(0, .) se do_relu == 1)
struct ConvBnParams {
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
    groups: u32,
    do_relu: u32, // 0 = Conv+BN (tipo 23), 1 = Conv+BN+Relu (tipo 24)
};

@group(0) @binding(0) var<uniform> params: ConvBnParams;
@group(0) @binding(1) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(2) var<storage, read> weights_tensor: array<f32>;
@group(0) @binding(3) var<storage, read> bn_a: array<f32>;
@group(0) @binding(4) var<storage, read> bn_b: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_tensor: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let co    = global_id.z;

    if (out_x >= params.out_w || out_y >= params.out_h || co >= params.c_out) {
        return;
    }

    var sum: f32 = 0.0;

    let cin_per_group = params.c_in / params.groups;
    let cout_per_group = params.c_out / params.groups;
    let group_id = co / cout_per_group;
    let ci_offset = group_id * cin_per_group;

    let in_base_h = i32(out_y * params.stride) - i32(params.padding);
    let in_base_w = i32(out_x * params.stride) - i32(params.padding);

    for (var ci: u32 = 0u; ci < cin_per_group; ci = ci + 1u) {
        let real_ci = ci_offset + ci;
        for (var ki: u32 = 0u; ki < params.kh; ki = ki + 1u) {
            for (var kj: u32 = 0u; kj < params.kw; kj = kj + 1u) {
                let in_h = in_base_h + i32(ki);
                let in_w = in_base_w + i32(kj);

                if (in_h >= 0 && in_h < i32(params.h_in) && in_w >= 0 && in_w < i32(params.w_in)) {
                    let input_idx = real_ci * (params.h_in * params.w_in)
                                  + u32(in_h) * params.w_in
                                  + u32(in_w);

                    // Peso no formato ONNX [c_out, c_in/groups, kh, kw]
                    let kernel_idx = co * (cin_per_group * params.kh * params.kw)
                                   + ci * (params.kh * params.kw)
                                   + ki * params.kw
                                   + kj;

                    sum = sum + input_tensor[input_idx] * weights_tensor[kernel_idx];
                }
            }
        }
    }

    var v = sum * bn_a[co] + bn_b[co];
    if (params.do_relu == 1u) {
        v = max(v, 0.0);
    }

    let out_idx = co * (params.out_h * params.out_w)
                + out_y * params.out_w
                + out_x;

    output_tensor[out_idx] = v;
}
