wit_bindgen::generate!({
    world: "bnworld-gpu",
    path: "wit",
    generate_all,
});

use exports::planner::bnworld::plan::Guest;
const KNAME: &str = "bn";
const SHADER: &str = include_str!("bn.wgsl");

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

struct W { s: webgpu::GpuBuffer, b: webgpu::GpuBuffer, m: webgpu::GpuBuffer, v: webgpu::GpuBuffer }
thread_local! { static WC: RefCell<HashMap<String, W>> = RefCell::new(HashMap::new()); }

impl Guest for Component {
    fn bn_load_weights(dev: &webgpu::GpuDevice, weight_id: String, scale: Vec<f32>, beta: Vec<f32>, mean: Vec<f32>, var: Vec<f32>) {
        let w = W { s: sbuf(dev, &scale), b: sbuf(dev, &beta), m: sbuf(dev, &mean), v: sbuf(dev, &var) };
        WC.with(|c| c.borrow_mut().insert(weight_id, w));
    }
    fn bn_r(dev: &webgpu::GpuDevice, weight_id: String, channels: u32, h_in: u32, w_in: u32, epsilon: f32, input: &webgpu::GpuBuffer) -> webgpu::GpuBuffer {
        let out = obuf(dev, (channels * h_in * w_in) as usize);
        let mut pb = u32s(&[channels, h_in, w_in]);
        pb.extend_from_slice(&epsilon.to_le_bytes());
        WC.with(|c| {
            let c = c.borrow();
            let w = c.get(&weight_id).unwrap_or_else(|| panic!("bn: {} sem load", weight_id));
            let pr = ubuf(dev, &pb);
            dispatch(dev, &[&pr, input, &w.s, &w.b, &w.m, &w.v, &out], ((w_in + 7) / 8, (h_in + 7) / 8, (channels + 3) / 4));
        });
        out
    }
}

export!(Component);
