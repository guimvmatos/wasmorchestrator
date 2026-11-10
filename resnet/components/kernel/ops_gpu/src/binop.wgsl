// P = [len_a, len_b, op]; out[i] = a[i] op b[i % len_b] (broadcast por módulo, igual aos kernels C)
@group(0) @binding(0) var<storage, read> P: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = g.x + g.y * nw.x * 256u;
    if (i >= P[0]) { return; }
    let x = a[i];
    let y = b[i % P[1]];
    var r: f32;
    switch P[2] {
        case 0u: { r = x + y; }
        case 1u: { r = x - y; }
        case 2u: { r = x * y; }
        default: { r = x / y; }
    }
    o[i] = r;
}
