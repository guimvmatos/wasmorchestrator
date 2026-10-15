wit_bindgen::generate!({
    world: "matmulworld-gpu",
    path: "wit",
    generate_all,
});

use exports::planner::matmulworld::plan::{Data, Guest};
use wasi::webgpu::webgpu;
use wasi_gfx_runtime::wasi_webgpu::sync_io;
use wit_bindgen::block_on;
use std::cell::RefCell;

struct Component;

struct GpuContext {
    device: webgpu::GpuDevice,
    pipeline: webgpu::GpuComputePipeline,
}

thread_local! {
    static GPU_CTX: RefCell<Option<GpuContext>> = RefCell::new(None);
}

fn build_gpu_context() -> GpuContext {
    let gpu = webgpu::get_gpu();
    let adapter_options = webgpu::GpuRequestAdapterOptions {
        feature_level: None,
        power_preference: Some(webgpu::GpuPowerPreference::HighPerformance),
        force_fallback_adapter: None,
        xr_compatible: None,
    };
    let adapter = block_on(gpu.request_adapter(Some(adapter_options))).unwrap();

    let info = adapter.info();
    eprintln!(
        "GPU selecionada (MatMul): vendor='{}' device='{}' architecture='{}'",
        info.vendor(),
        info.device(),
        info.architecture()
    );

    let device = block_on(adapter.request_device(None)).unwrap();

    let cs_module = device.create_shader_module(&webgpu::GpuShaderModuleDescriptor {
        label: Some("MatMul Shader".to_string()),
        code: include_str!("matmul.wgsl").to_string(),
        compilation_hints: None,
    });

    let pipeline = device.create_compute_pipeline(webgpu::GpuComputePipelineDescriptor {
        label: Some("MatMul Pipeline".to_string()),
        layout: webgpu::GpuLayoutMode::Auto,
        compute: webgpu::GpuProgrammableStage {
            module: &cs_module,
            entry_point: Some("main".to_string()),
            constants: None,
        },
    });

    GpuContext { device, pipeline }
}

impl Guest for Component {
    fn matmul(lhs: Vec<f32>, rhs: Vec<f32>, lhs_shape: Vec<u32>, rhs_shape: Vec<u32>, result_container: Data) -> Data {
        let output = execute_matmul_gpu(&lhs, &rhs, &lhs_shape, &rhs_shape);

        Data {
            input: vec![],
            output,
            reply_to: result_container.reply_to,
            kernels: vec![],
            current_kernel: result_container.current_kernel,
            request: result_container.request,
            from_step: result_container.from_step,
        }
    }
}

// Espelha EXATAMENTE a lógica de broadcasting de batch do kernel CPU em C
// (kernel/matmul/matmul.c: compute_batch_matmul) — as duas últimas
// dimensões de cada shape são as dimensões de matriz (M,K) e (K,N); o
// resto é batch e faz broadcast alinhado pela direita (dimensão 1 contra
// qualquer tamanho). Devolve M, K, N, total_batch e os offsets (em
// elementos) de cada matriz M*K/K*N dentro de lhs/rhs por índice de batch,
// prontos para subir como buffers de leitura no shader (sem lógica de
// indexação dinâmica dentro do WGSL).
fn compute_batch_layout(lhs_shape: &[u32], rhs_shape: &[u32]) -> (u32, u32, u32, Vec<u32>, Vec<u32>) {
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();
    let lhs_batch_rank = lhs_rank - 2;
    let rhs_batch_rank = rhs_rank - 2;
    let batch_rank = lhs_batch_rank.max(rhs_batch_rank);

    let dim_at = |shape: &[u32], batch_rank_of_shape: usize, i: usize| -> u32 {
        // i = índice em batch_shape (0 = mais lento/esquerda, batch_rank-1 =
        // mais rápido/direita, perto de M/K). Alinhamento pela direita
        // (regra do ONNX/numpy): dimensões que faltam contam como 1.
        let li = batch_rank_of_shape as isize - batch_rank as isize + i as isize;
        if li >= 0 { shape[li as usize] } else { 1 }
    };

    let mut batch_shape = vec![1u32; batch_rank];
    for i in 0..batch_rank {
        let ld = dim_at(lhs_shape, lhs_batch_rank, i);
        let rd = dim_at(rhs_shape, rhs_batch_rank, i);
        batch_shape[i] = ld.max(rd);
    }

    let total_batch: u32 = batch_shape.iter().product::<u32>().max(1);

    let m = lhs_shape[lhs_rank - 2];
    let k = lhs_shape[lhs_rank - 1];
    let n = rhs_shape[rhs_rank - 1];

    let lhs_mat_size = m * k;
    let rhs_mat_size = k * n;

    // Stride (em elementos) de cada dimensão de batch — dimensão broadcast
    // (tamanho 1 sem correspondente real no outro tensor) tem stride 0.
    let mut lhs_bstride = vec![0u32; batch_rank];
    let mut rhs_bstride = vec![0u32; batch_rank];
    {
        let mut lhs_running = lhs_mat_size;
        let mut rhs_running = rhs_mat_size;
        for idx in 0..batch_rank {
            let i = batch_rank - 1 - idx;
            let ld = dim_at(lhs_shape, lhs_batch_rank, i);
            let rd = dim_at(rhs_shape, rhs_batch_rank, i);
            lhs_bstride[i] = if ld == 1 { 0 } else { lhs_running };
            rhs_bstride[i] = if rd == 1 { 0 } else { rhs_running };
            if ld != 1 {
                lhs_running *= ld;
            }
            if rd != 1 {
                rhs_running *= rd;
            }
        }
    }

    let mut lhs_offsets = vec![0u32; total_batch as usize];
    let mut rhs_offsets = vec![0u32; total_batch as usize];
    for b in 0..total_batch {
        let mut rem = b;
        let mut lhs_off = 0u32;
        let mut rhs_off = 0u32;
        for idx in 0..batch_rank {
            let i = batch_rank - 1 - idx;
            let dim = batch_shape[i];
            let coord = rem % dim;
            rem /= dim;
            lhs_off += coord * lhs_bstride[i];
            rhs_off += coord * rhs_bstride[i];
        }
        lhs_offsets[b as usize] = lhs_off;
        rhs_offsets[b as usize] = rhs_off;
    }

    (m, k, n, lhs_offsets, rhs_offsets)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DimsUniform {
    m: u32,
    k: u32,
    n: u32,
    total_batch: u32,
}

fn execute_matmul_gpu(lhs: &[f32], rhs: &[f32], lhs_shape: &[u32], rhs_shape: &[u32]) -> Vec<f32> {
    let (m, k, n, lhs_offsets, rhs_offsets) = compute_batch_layout(lhs_shape, rhs_shape);
    let total_batch = lhs_offsets.len() as u32;
    let out_len = (total_batch * m * n) as usize;

    GPU_CTX.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(build_gpu_context());
        }
        let ctx_ref = cell.borrow();
        let ctx = ctx_ref.as_ref().unwrap();
        let device = &ctx.device;
        let compute_pipeline = &ctx.pipeline;

        let lhs_bytes = bytemuck::cast_slice(lhs);
        let lhs_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
            label: Some("MatMul LHS Buffer".to_string()),
            size: lhs_bytes.len() as webgpu::GpuSize64,
            usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_DST,
            mapped_at_creation: Some(true),
        });
        lhs_buffer.get_mapped_range_set_with_copy(lhs_bytes, None, None).unwrap();
        lhs_buffer.unmap().unwrap();

        let rhs_bytes = bytemuck::cast_slice(rhs);
        let rhs_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
            label: Some("MatMul RHS Buffer".to_string()),
            size: rhs_bytes.len() as webgpu::GpuSize64,
            usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_DST,
            mapped_at_creation: Some(true),
        });
        rhs_buffer.get_mapped_range_set_with_copy(rhs_bytes, None, None).unwrap();
        rhs_buffer.unmap().unwrap();

        let lhs_off_bytes = bytemuck::cast_slice(&lhs_offsets);
        let lhs_off_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
            label: Some("MatMul LHS Offsets Buffer".to_string()),
            size: lhs_off_bytes.len() as webgpu::GpuSize64,
            usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_DST,
            mapped_at_creation: Some(true),
        });
        lhs_off_buffer.get_mapped_range_set_with_copy(lhs_off_bytes, None, None).unwrap();
        lhs_off_buffer.unmap().unwrap();

        let rhs_off_bytes = bytemuck::cast_slice(&rhs_offsets);
        let rhs_off_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
            label: Some("MatMul RHS Offsets Buffer".to_string()),
            size: rhs_off_bytes.len() as webgpu::GpuSize64,
            usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_DST,
            mapped_at_creation: Some(true),
        });
        rhs_off_buffer.get_mapped_range_set_with_copy(rhs_off_bytes, None, None).unwrap();
        rhs_off_buffer.unmap().unwrap();

        let dims = DimsUniform { m, k, n, total_batch };
        let dims_bytes = bytemuck::bytes_of(&dims);
        let dims_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
            label: Some("MatMul Dims Buffer".to_string()),
            size: dims_bytes.len() as webgpu::GpuSize64,
            usage: webgpu::GpuBufferUsage::UNIFORM | webgpu::GpuBufferUsage::COPY_DST,
            mapped_at_creation: Some(true),
        });
        dims_buffer.get_mapped_range_set_with_copy(dims_bytes, None, None).unwrap();
        dims_buffer.unmap().unwrap();

        let output_byte_size = (out_len * std::mem::size_of::<f32>()) as webgpu::GpuSize64;
        let output_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
            label: Some("MatMul Output Buffer".to_string()),
            size: output_byte_size,
            usage: webgpu::GpuBufferUsage::STORAGE
                | webgpu::GpuBufferUsage::COPY_SRC
                | webgpu::GpuBufferUsage::COPY_DST,
            mapped_at_creation: None,
        });

        let staging_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
            label: Some("MatMul Staging Buffer".to_string()),
            size: output_byte_size,
            usage: webgpu::GpuBufferUsage::MAP_READ | webgpu::GpuBufferUsage::COPY_DST,
            mapped_at_creation: None,
        });

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&webgpu::GpuBindGroupDescriptor {
            label: Some("MatMul Bind Group".to_string()),
            layout: &bind_group_layout,
            entries: vec![
                webgpu::GpuBindGroupEntry {
                    binding: 0,
                    resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding {
                        buffer: &lhs_buffer,
                        offset: Some(0),
                        size: None,
                    }),
                },
                webgpu::GpuBindGroupEntry {
                    binding: 1,
                    resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding {
                        buffer: &rhs_buffer,
                        offset: Some(0),
                        size: None,
                    }),
                },
                webgpu::GpuBindGroupEntry {
                    binding: 2,
                    resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding {
                        buffer: &lhs_off_buffer,
                        offset: Some(0),
                        size: None,
                    }),
                },
                webgpu::GpuBindGroupEntry {
                    binding: 3,
                    resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding {
                        buffer: &rhs_off_buffer,
                        offset: Some(0),
                        size: None,
                    }),
                },
                webgpu::GpuBindGroupEntry {
                    binding: 4,
                    resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding {
                        buffer: &dims_buffer,
                        offset: Some(0),
                        size: None,
                    }),
                },
                webgpu::GpuBindGroupEntry {
                    binding: 5,
                    resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding {
                        buffer: &output_buffer,
                        offset: Some(0),
                        size: None,
                    }),
                },
            ],
        });

        let encoder = device.create_command_encoder(Some(&webgpu::GpuCommandEncoderDescriptor { label: None }));
        {
            let cpass = encoder.begin_compute_pass(None);
            cpass.set_pipeline(compute_pipeline);
            cpass
                .set_bind_group(0, Some(&bind_group), None, None, None)
                .unwrap();

            let total_threads = total_batch * m * n;
            let workgroups = (total_threads + 63) / 64;
            cpass.dispatch_workgroups(workgroups.max(1), Some(1), Some(1));
            cpass.end();
        }

        encoder.copy_buffer_to_buffer(&output_buffer, None, &staging_buffer, None, None);
        device.queue().submit(&[&encoder.finish(None)]);

        let data = sync_io::map_and_read_sync(
            &staging_buffer,
            webgpu::GpuMapMode::READ,
            Some(0),
            None,
        )
        .expect("map_and_read_sync falhou no host (MatMul)");

        if !data.is_empty() {
            bytemuck::cast_slice(&data).to_vec()
        } else {
            vec![0.0f32; out_len]
        }
    })
}

export!(Component);
