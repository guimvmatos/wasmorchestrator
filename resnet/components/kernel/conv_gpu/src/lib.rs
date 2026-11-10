wit_bindgen::generate!({
    world: "convworld-gpu",
    path: "wit",
    generate_all,
});

use exports::planner::convworld::plan::Guest;
const KNAME: &str = "conv";
const SHADER: &str = include_str!("conv.wgsl");

// ---- 10/11 (4c): via residente na GPU ------------------------------------
// O device é do GRUPO (passado como borrow em toda chamada). A entrada chega
// como handle de um buffer que já está na GPU; a saída é um buffer novo,
// devolvido como handle (own). Nada de readback aqui: quem baixa para a CPU,
// quando precisa, é o grupo. Os comandos vão na fila do mesmo device, então a
// ordem entre kernels é garantida pela própria fila.
use std::cell::RefCell;
#[allow(unused_imports)]
use std::collections::HashMap;
use wasi::webgpu::webgpu;

thread_local! {
    static PIPE: RefCell<Option<webgpu::GpuComputePipeline>> = RefCell::new(None);
}

fn with_pipeline<R>(dev: &webgpu::GpuDevice, f: impl FnOnce(&webgpu::GpuComputePipeline) -> R) -> R {
    PIPE.with(|p| {
        if p.borrow().is_none() {
            let m = dev.create_shader_module(&webgpu::GpuShaderModuleDescriptor {
                label: Some(KNAME.to_string()),
                code: SHADER.to_string(),
                compilation_hints: None,
            });
            let pl = dev.create_compute_pipeline(webgpu::GpuComputePipelineDescriptor {
                label: Some(KNAME.to_string()),
                layout: webgpu::GpuLayoutMode::Auto,
                compute: webgpu::GpuProgrammableStage { module: &m, entry_point: Some("main".to_string()), constants: None },
            });
            *p.borrow_mut() = Some(pl);
        }
        f(p.borrow().as_ref().unwrap())
    })
}

fn ubuf(dev: &webgpu::GpuDevice, bytes: &[u8]) -> webgpu::GpuBuffer {
    let b = dev.create_buffer(&webgpu::GpuBufferDescriptor {
        label: None, size: bytes.len().max(16) as u64,
        usage: webgpu::GpuBufferUsage::UNIFORM | webgpu::GpuBufferUsage::COPY_DST,
        mapped_at_creation: Some(true),
    });
    let mut padded = bytes.to_vec();
    padded.resize(bytes.len().max(16), 0);
    b.get_mapped_range_set_with_copy(&padded, None, None).unwrap();
    b.unmap().unwrap();
    b
}

#[allow(dead_code)]
fn sbuf(dev: &webgpu::GpuDevice, data: &[f32]) -> webgpu::GpuBuffer {
    let src: &[f32] = if data.is_empty() { &[0.0f32] } else { data };
    let bytes = bytemuck::cast_slice(src);
    let b = dev.create_buffer(&webgpu::GpuBufferDescriptor {
        label: None, size: bytes.len() as u64,
        usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_DST,
        mapped_at_creation: Some(true),
    });
    b.get_mapped_range_set_with_copy(bytes, None, None).unwrap();
    b.unmap().unwrap();
    b
}

fn obuf(dev: &webgpu::GpuDevice, n: usize) -> webgpu::GpuBuffer {
    dev.create_buffer(&webgpu::GpuBufferDescriptor {
        label: None, size: (n.max(1) * 4) as u64,
        usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_SRC | webgpu::GpuBufferUsage::COPY_DST,
        mapped_at_creation: None,
    })
}

fn entry<'a>(i: u32, b: &'a webgpu::GpuBuffer) -> webgpu::GpuBindGroupEntry<'a> {
    webgpu::GpuBindGroupEntry { binding: i, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: b, offset: Some(0), size: None }) }
}

/// Liga os buffers nos bindings 0..n (na ordem), despacha e envia. Sem espera.
fn dispatch(dev: &webgpu::GpuDevice, bufs: &[&webgpu::GpuBuffer], wg: (u32, u32, u32)) {
    with_pipeline(dev, |pl| {
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

fn u32s(v: &[u32]) -> Vec<u8> { bytemuck::cast_slice(v).to_vec() }

struct Component;

struct W { w: webgpu::GpuBuffer, b: webgpu::GpuBuffer, has_bias: u32 }
thread_local! { static WC: RefCell<HashMap<String, W>> = RefCell::new(HashMap::new()); }

impl Guest for Component {
    fn conv_load_weights(dev: &webgpu::GpuDevice, weight_id: String, weights: Vec<f32>, bias: Vec<f32>) {
        let has_bias = if bias.is_empty() { 0 } else { 1 };
        let w = W { w: sbuf(dev, &weights), b: sbuf(dev, &bias), has_bias };
        WC.with(|c| c.borrow_mut().insert(weight_id, w));
    }
    fn conv_r(dev: &webgpu::GpuDevice, weight_id: String, c_out: u32, c_in: u32, h_in: u32, w_in: u32, kh: u32, kw: u32, stride: u32, padding: u32, groups: u32, input: &webgpu::GpuBuffer) -> webgpu::GpuBuffer {
        let groups = if groups == 0 { 1 } else { groups };
        let out_h = ((h_in + 2 * padding - kh) / stride) + 1;
        let out_w = ((w_in + 2 * padding - kw) / stride) + 1;
        let out = obuf(dev, (c_out * out_h * out_w) as usize);
        WC.with(|c| {
            let c = c.borrow();
            let w = c.get(&weight_id).unwrap_or_else(|| panic!("conv: {} sem load", weight_id));
            let pr = ubuf(dev, &u32s(&[c_out, c_in, h_in, w_in, kh, kw, stride, padding, out_h, out_w, groups, w.has_bias]));
            dispatch(dev, &[&pr, input, &w.w, &w.b, &out], ((out_w + 7) / 8, (out_h + 7) / 8, (c_out + 3) / 4));
        });
        out
    }
}

export!(Component);
