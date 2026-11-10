// Transpose e Slice: P = [total, rank, base(i32), out_stride[6], in_step[6](i32)]
// in_linear = base + sum_d coord_d * in_step_d (in_step pode ser negativo: Slice com step < 0)
@group(0) @binding(0) var<storage, read> P: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = g.x + g.y * nw.x * 256u;
    if (i >= P[0]) { return; }
    var rem = i;
    var idx: i32 = bitcast<i32>(P[2]);
    for (var d = 0u; d < P[1]; d = d + 1u) {
        let os = P[3u + d];
        let c = rem / os;
        rem = rem % os;
        idx = idx + i32(c) * bitcast<i32>(P[9u + d]);
    }
    o[i] = a[u32(idx)];
}
