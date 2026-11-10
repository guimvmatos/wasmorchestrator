// 10/11 (4c, parte 2): operadores leves na via residente da GPU.
// Um componente com vários shaders (Sigmoid, SiLU, Add/Sub/Mul/Div com
// broadcast por módulo, cópia com passo para Concat/Split, Transpose, Slice,
// Resize nearest, Softmax e MatMul em lote). Mesmo contrato dos kernels da
// via: device do grupo, entrada como handle de buffer na GPU, saída como
// buffer novo; sem readback aqui. A semântica segue os kernels em C.

wit_bindgen::generate!({
    world: "opsgpuworld-gpu",
    path: "wit",
    generate_all,
});

use exports::planner::opsgpuworld::plan::Guest;
use std::cell::RefCell;
use std::collections::HashMap;
use wasi::webgpu::webgpu;

struct Component;

thread_local! {
    static PIPES: RefCell<HashMap<&'static str, webgpu::GpuComputePipeline>> = RefCell::new(HashMap::new());
}

fn shader(name: &'static str) -> &'static str {
    match name {
        "unary" => include_str!("unary.wgsl"),
        "binop" => include_str!("binop.wgsl"),
        "copy" => include_str!("copy.wgsl"),
        "gather" => include_str!("gather.wgsl"),
        "resize" => include_str!("resize.wgsl"),
        "softmax" => include_str!("softmax.wgsl"),
        "matmul" => include_str!("matmul.wgsl"),
        _ => unreachable!(),
    }
}

fn with_pipeline<R>(dev: &webgpu::GpuDevice, name: &'static str, f: impl FnOnce(&webgpu::GpuComputePipeline) -> R) -> R {
    PIPES.with(|p| {
        if !p.borrow().contains_key(name) {
            let m = dev.create_shader_module(&webgpu::GpuShaderModuleDescriptor {
                label: Some(name.to_string()),
                code: shader(name).to_string(),
                compilation_hints: None,
            });
            let pl = dev.create_compute_pipeline(webgpu::GpuComputePipelineDescriptor {
                label: Some(name.to_string()),
                layout: webgpu::GpuLayoutMode::Auto,
                compute: webgpu::GpuProgrammableStage { module: &m, entry_point: Some("main".to_string()), constants: None },
            });
            p.borrow_mut().insert(name, pl);
        }
        f(p.borrow().get(name).unwrap())
    })
}

/// Parâmetros como storage buffer de u32 (evita as regras de alinhamento de
/// arrays em uniform).
fn pbuf(dev: &webgpu::GpuDevice, v: &[u32]) -> webgpu::GpuBuffer {
    let src: &[u32] = if v.is_empty() { &[0u32] } else { v };
    let bytes: &[u8] = bytemuck::cast_slice(src);
    let b = dev.create_buffer(&webgpu::GpuBufferDescriptor {
        label: None,
        size: bytes.len() as u64,
        usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_DST,
        mapped_at_creation: Some(true),
    });
    b.get_mapped_range_set_with_copy(bytes, None, None).unwrap();
    b.unmap().unwrap();
    b
}

fn obuf(dev: &webgpu::GpuDevice, n: usize) -> webgpu::GpuBuffer {
    dev.create_buffer(&webgpu::GpuBufferDescriptor {
        label: None,
        size: (n.max(1) * 4) as u64,
        usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_SRC | webgpu::GpuBufferUsage::COPY_DST,
        mapped_at_creation: None,
    })
}

fn entry<'a>(i: u32, b: &'a webgpu::GpuBuffer) -> webgpu::GpuBindGroupEntry<'a> {
    webgpu::GpuBindGroupEntry {
        binding: i,
        resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: b, offset: Some(0), size: None }),
    }
}

fn dispatch(dev: &webgpu::GpuDevice, name: &'static str, bufs: &[&webgpu::GpuBuffer], wg: (u32, u32, u32)) {
    with_pipeline(dev, name, |pl| {
        let layout = pl.get_bind_group_layout(0);
        let entries: Vec<webgpu::GpuBindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| entry(i as u32, b)).collect();
        let bg = dev.create_bind_group(&webgpu::GpuBindGroupDescriptor { label: None, layout: &layout, entries });
        let enc = dev.create_command_encoder(None);
        {
            let c = enc.begin_compute_pass(None);
            c.set_pipeline(pl);
            c.set_bind_group(0, Some(&bg), None, None, None).unwrap();
            c.dispatch_workgroups(wg.0, Some(wg.1), Some(wg.2));
            c.end();
        }
        dev.queue().submit(&[&enc.finish(None)]);
    })
}

/// Grade 1-D de grupos de 256 threads, dobrada em 2-D quando passa de 65535.
fn grid(n: usize) -> (u32, u32, u32) {
    let groups = ((n + 255) / 256).max(1) as u32;
    let gx = groups.min(65535);
    (gx, (groups + gx - 1) / gx, 1)
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let r = shape.len();
    let mut s = vec![1usize; r];
    for i in (1..r).rev() {
        s[i - 1] = s[i] * shape[i];
    }
    s
}

/// Gather genérico (Transpose/Slice): out[i] = in[base + sum coord_d * step_d].
fn gather(dev: &webgpu::GpuDevice, out_shape: &[usize], base: i64, in_step: &[i64], input: &webgpu::GpuBuffer) -> (webgpu::GpuBuffer, u32) {
    let rank = out_shape.len();
    assert!(rank >= 1 && rank <= 6, "opsgpu gather: rank {} fora de 1..6", rank);
    let total: usize = out_shape.iter().product();
    let os = strides(out_shape);
    let mut p = vec![0u32; 15];
    p[0] = total as u32;
    p[1] = rank as u32;
    p[2] = (base as i32) as u32;
    for d in 0..rank {
        p[3 + d] = os[d].max(1) as u32;
        p[9 + d] = (in_step[d] as i32) as u32;
    }
    let out = obuf(dev, total);
    if total > 0 {
        let pb = pbuf(dev, &p);
        dispatch(dev, "gather", &[&pb, input, &out], grid(total));
    }
    (out, total as u32)
}

impl Guest for Component {
    fn unary_r(dev: &webgpu::GpuDevice, op: u32, n: u32, input: &webgpu::GpuBuffer) -> webgpu::GpuBuffer {
        let out = obuf(dev, n as usize);
        let pb = pbuf(dev, &[n, op]);
        dispatch(dev, "unary", &[&pb, input, &out], grid(n as usize));
        out
    }

    fn binop_r(dev: &webgpu::GpuDevice, op: u32, len_a: u32, len_b: u32, a: &webgpu::GpuBuffer, b: &webgpu::GpuBuffer) -> webgpu::GpuBuffer {
        let out = obuf(dev, len_a as usize);
        let pb = pbuf(dev, &[len_a, len_b.max(1), op]);
        dispatch(dev, "binop", &[&pb, a, b, &out], grid(len_a as usize));
        out
    }

    fn copy_strided(dev: &webgpu::GpuDevice, outer: u32, chunk: u32, src_stride: u32, src_off: u32, dst_stride: u32, dst_off: u32, src: &webgpu::GpuBuffer, dst: &webgpu::GpuBuffer) {
        let n = (outer as usize) * (chunk as usize);
        if n == 0 {
            return;
        }
        let pb = pbuf(dev, &[outer, chunk, src_stride, src_off, dst_stride, dst_off]);
        dispatch(dev, "copy", &[&pb, src, dst], grid(n));
    }

    fn transpose_r(dev: &webgpu::GpuDevice, shape: Vec<u32>, perm: Vec<u32>, input: &webgpu::GpuBuffer) -> webgpu::GpuBuffer {
        let sh: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let ins = strides(&sh);
        let out_shape: Vec<usize> = perm.iter().map(|&p| sh[p as usize]).collect();
        let step: Vec<i64> = perm.iter().map(|&p| ins[p as usize] as i64).collect();
        gather(dev, &out_shape, 0, &step, input).0
    }

    fn slice_r(dev: &webgpu::GpuDevice, shape: Vec<u32>, starts: Vec<i32>, ends: Vec<i32>, axes: Vec<u32>, steps: Vec<i32>, input: &webgpu::GpuBuffer) -> (webgpu::GpuBuffer, u32) {
        // Mesma normalização do slice.c.
        let rank = shape.len();
        let mut begin: Vec<i64> = vec![0; rank];
        let mut end: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let mut stp: Vec<i64> = vec![1; rank];
        for k in 0..axes.len() {
            let axis = axes[k] as usize;
            let dim = shape[axis] as i64;
            let mut s = starts[k] as i64;
            let mut e = ends[k] as i64;
            let mut st = if k < steps.len() { steps[k] as i64 } else { 1 };
            if st == 0 {
                st = 1;
            }
            if st > 0 {
                if s < 0 { s += dim; }
                s = s.clamp(0, dim);
                if e < 0 { e += dim; }
                e = e.clamp(0, dim);
            } else {
                if s < 0 { s += dim; }
                s = s.clamp(-1, dim - 1);
                if e < 0 { e += dim; }
                e = e.clamp(-1, dim - 1);
            }
            begin[axis] = s;
            end[axis] = e;
            stp[axis] = st;
        }
        let mut out_shape = vec![0usize; rank];
        for i in 0..rank {
            let diff = if stp[i] > 0 { end[i] - begin[i] } else { begin[i] - end[i] };
            let sa = stp[i].abs();
            out_shape[i] = if diff > 0 { ((diff + sa - 1) / sa) as usize } else { 0 };
        }
        let sh: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let ins = strides(&sh);
        let base: i64 = (0..rank).map(|i| begin[i] * ins[i] as i64).sum();
        let step: Vec<i64> = (0..rank).map(|i| stp[i] * ins[i] as i64).collect();
        gather(dev, &out_shape, base, &step, input)
    }

    fn resize_r(dev: &webgpu::GpuDevice, channels: u32, h_in: u32, w_in: u32, scale_h: f32, scale_w: f32, input: &webgpu::GpuBuffer) -> (webgpu::GpuBuffer, u32) {
        let h_out = ((h_in as f32 * scale_h + 0.5f32) as u32).max(1);
        let w_out = ((w_in as f32 * scale_w + 0.5f32) as u32).max(1);
        let total = (channels * h_out * w_out) as usize;
        let out = obuf(dev, total);
        let pb = pbuf(dev, &[channels, h_in, w_in, h_out, w_out, scale_h.to_bits(), scale_w.to_bits()]);
        dispatch(dev, "resize", &[&pb, input, &out], grid(total));
        (out, total as u32)
    }

    fn softmax_r(dev: &webgpu::GpuDevice, shape: Vec<u32>, axis: u32, input: &webgpu::GpuBuffer) -> webgpu::GpuBuffer {
        let a = axis as usize;
        let outer: usize = shape[..a].iter().map(|&d| d as usize).product();
        let ax = shape[a] as usize;
        let inner: usize = shape[a + 1..].iter().map(|&d| d as usize).product();
        let out = obuf(dev, outer * ax * inner);
        let pb = pbuf(dev, &[outer as u32, ax as u32, inner as u32]);
        dispatch(dev, "softmax", &[&pb, input, &out], grid(outer * inner));
        out
    }

    fn matmul_r(dev: &webgpu::GpuDevice, lhs_shape: Vec<u32>, rhs_shape: Vec<u32>, lhs: &webgpu::GpuBuffer, rhs: &webgpu::GpuBuffer) -> (webgpu::GpuBuffer, u32) {
        // Mesmo broadcast de lotes do matmul.c.
        let lr = lhs_shape.len();
        let rr = rhs_shape.len();
        let lb = lr - 2;
        let rb = rr - 2;
        let br = lb.max(rb);
        let dim_l = |i: usize| -> usize { let li = lb as i64 - br as i64 + i as i64; if li >= 0 { lhs_shape[li as usize] as usize } else { 1 } };
        let dim_r = |i: usize| -> usize { let ri = rb as i64 - br as i64 + i as i64; if ri >= 0 { rhs_shape[ri as usize] as usize } else { 1 } };
        let bshape: Vec<usize> = (0..br).map(|i| dim_l(i).max(dim_r(i))).collect();
        let total_batch: usize = bshape.iter().product();
        let m = lhs_shape[lr - 2] as usize;
        let k = lhs_shape[lr - 1] as usize;
        let n = rhs_shape[rr - 1] as usize;
        let mut lst = vec![0usize; br];
        let mut rst = vec![0usize; br];
        let (mut lrun, mut rrun) = (m * k, k * n);
        for i in (0..br).rev() {
            let (ld, rd) = (dim_l(i), dim_r(i));
            lst[i] = if ld == 1 { 0 } else { lrun };
            rst[i] = if rd == 1 { 0 } else { rrun };
            if ld != 1 { lrun *= ld; }
            if rd != 1 { rrun *= rd; }
        }
        let mut offs: Vec<u32> = Vec::with_capacity(total_batch * 2);
        for b in 0..total_batch {
            let mut rem = b;
            let (mut lo, mut ro) = (0usize, 0usize);
            for i in (0..br).rev() {
                let c = rem % bshape[i];
                rem /= bshape[i];
                lo += c * lst[i];
                ro += c * rst[i];
            }
            offs.push(lo as u32);
            offs.push(ro as u32);
        }
        let total = total_batch * m * n;
        let out = obuf(dev, total);
        let pb = pbuf(dev, &[m as u32, k as u32, n as u32, total_batch as u32]);
        let ob = pbuf(dev, &offs);
        dispatch(dev, "matmul", &[&pb, &ob, lhs, rhs, &out], (((n + 7) / 8) as u32, ((m + 7) / 8) as u32, total_batch as u32));
        (out, total as u32)
    }
}

export!(Component);
