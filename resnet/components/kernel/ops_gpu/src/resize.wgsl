// Nearest (resize.c): P = [c, h_in, w_in, h_out, w_out, scale_h(f32), scale_w(f32)]
@group(0) @binding(0) var<storage, read> P: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = g.x + g.y * nw.x * 256u;
    let h_in = P[1]; let w_in = P[2]; let h_out = P[3]; let w_out = P[4];
    if (i >= P[0] * h_out * w_out) { return; }
    let ox = i % w_out;
    let t = i / w_out;
    let oy = t % h_out;
    let c = t / h_out;
    var iy = u32(f32(oy) / bitcast<f32>(P[5]));
    if (iy >= h_in) { iy = h_in - 1u; }
    var ix = u32(f32(ox) / bitcast<f32>(P[6]));
    if (ix >= w_in) { ix = w_in - 1u; }
    o[i] = a[c * h_in * w_in + iy * w_in + ix];
}
