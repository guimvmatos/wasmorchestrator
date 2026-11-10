// P = [outer, chunk, src_stride, src_off, dst_stride, dst_off]
@group(0) @binding(0) var<storage, read> P: array<u32>;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> d: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = g.x + g.y * nw.x * 256u;
    let chunk = P[1];
    if (i >= P[0] * chunk) { return; }
    let r = i / chunk;
    let j = i % chunk;
    d[P[5] + r * P[4] + j] = s[P[3] + r * P[2] + j];
}
