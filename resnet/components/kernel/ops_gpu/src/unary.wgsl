// P = [n, op]; op 0 = sigmoid, 1 = silu (mesma fórmula do sigmoid.c/silu.c)
@group(0) @binding(0) var<storage, read> P: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = g.x + g.y * nw.x * 256u;
    if (i >= P[0]) { return; }
    let x = a[i];
    let s = 1.0 / (1.0 + exp(-x));
    if (P[1] == 0u) { o[i] = s; } else { o[i] = x * s; }
}
