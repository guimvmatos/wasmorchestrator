// MatMul em lote com broadcast (matmul.c): P = [M, K, N, batch]; offs = (lhs_off, rhs_off) por lote
@group(0) @binding(0) var<storage, read> P: array<u32>;
@group(0) @binding(1) var<storage, read> offs: array<u32>;
@group(0) @binding(2) var<storage, read> l: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let M = P[0]; let K = P[1]; let N = P[2];
    let n = g.x; let m = g.y; let b = g.z;
    if (n >= N || m >= M || b >= P[3]) { return; }
    let lo = offs[2u * b] + m * K;
    let ro = offs[2u * b + 1u] + n;
    var acc = 0.0;
    for (var k = 0u; k < K; k = k + 1u) { acc = acc + l[lo + k] * r[ro + k * N]; }
    o[b * M * N + m * N + n] = acc;
}
