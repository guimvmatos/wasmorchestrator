// Softmax estável num eixo (softmax.c): P = [outer, axis, inner]; uma thread por (outer, inner)
@group(0) @binding(0) var<storage, read> P: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = g.x + g.y * nw.x * 256u;
    let ax = P[1]; let inner = P[2];
    if (i >= P[0] * inner) { return; }
    let base = (i / inner) * ax * inner + (i % inner);
    var m = a[base];
    for (var k = 1u; k < ax; k = k + 1u) { m = max(m, a[base + k * inner]); }
    var s = 0.0;
    for (var k = 0u; k < ax; k = k + 1u) {
        let e = exp(a[base + k * inner] - m);
        o[base + k * inner] = e;
        s = s + e;
    }
    for (var k = 0u; k < ax; k = k + 1u) { o[base + k * inner] = o[base + k * inner] / s; }
}
