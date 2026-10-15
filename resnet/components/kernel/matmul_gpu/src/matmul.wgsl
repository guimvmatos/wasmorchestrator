struct Dims {
    m: u32,
    k: u32,
    n: u32,
    total_batch: u32,
};

@group(0) @binding(0) var<storage, read> lhs_tensor: array<f32>;
@group(0) @binding(1) var<storage, read> rhs_tensor: array<f32>;
@group(0) @binding(2) var<storage, read> lhs_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> rhs_offsets: array<u32>;
@group(0) @binding(4) var<uniform> dims: Dims;
@group(0) @binding(5) var<storage, read_write> output_tensor: array<f32>;

// Um thread por elemento de saida (batch, m, n). lhs_offsets[b]/
// rhs_offsets[b] ja vem pre-calculados do lado host (mesma logica de
// broadcasting do kernel CPU em C, ver kernel/matmul/matmul.c) — o shader
// so soma o produto escalar da linha m de lhs pela coluna n de rhs.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let mn = dims.m * dims.n;
    let total = dims.total_batch * mn;
    if (idx >= total) {
        return;
    }

    let b = idx / mn;
    let rem = idx % mn;
    let m = rem / dims.n;
    let n = rem % dims.n;

    let lhs_base = lhs_offsets[b];
    let rhs_base = rhs_offsets[b];

    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < dims.k; k = k + 1u) {
        acc = acc + lhs_tensor[lhs_base + m * dims.k + k] * rhs_tensor[rhs_base + k * dims.n + n];
    }
    output_tensor[idx] = acc;
}
