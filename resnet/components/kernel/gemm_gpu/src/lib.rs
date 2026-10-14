wit_bindgen::generate!({
    world: "gemmworld-gpu",
    path: "wit",
    generate_all,
});

use exports::planner::gemmworld::plan::{Data, Guest};
use wasi::webgpu::webgpu;
use wasi_gfx_runtime::wasi_webgpu::sync_io;
use wit_bindgen::block_on;
use std::cell::RefCell;
use std::collections::HashMap;

struct Component;

struct GpuContext {
    device: webgpu::GpuDevice,
    pipeline: webgpu::GpuComputePipeline,
}

// Buffers GPU já preenchidos para um weight-id específico — reaproveitados
// em toda chamada de `gemm` subsequente, sem recriar nem reenviar os pesos.
struct CachedWeights {
    weights_buffer: webgpu::GpuBuffer,
    bias_buffer: webgpu::GpuBuffer,
}

thread_local! {
    static GPU_CTX: RefCell<Option<GpuContext>> = RefCell::new(None);
    static WEIGHTS_CACHE: RefCell<HashMap<String, CachedWeights>> = RefCell::new(HashMap::new());
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
        "GPU selecionada (Gemm): vendor='{}' device='{}' architecture='{}'",
        info.vendor(),
        info.device(),
        info.architecture()
    );

    let limits = adapter.limits();
    let max_storage = limits.max_storage_buffer_binding_size();
    let max_buffer = limits.max_buffer_size();
    eprintln!(
        "Limites do adapter (Gemm): max_storage_buffer_binding_size={} max_buffer_size={}",
        max_storage, max_buffer
    );

    let required_limits = webgpu::RecordOptionGpuSize64::new();
    required_limits.add("maxStorageBufferBindingSize", Some(max_storage));
    required_limits.add("maxBufferSize", Some(max_buffer));

    let device_descriptor = webgpu::GpuDeviceDescriptor {
        required_features: None,
        required_limits: Some(required_limits),
        default_queue: None,
        label: Some("Gemm Device".to_string()),
    };

    let device = block_on(adapter.request_device(Some(device_descriptor))).unwrap();

    let cs_module = device.create_shader_module(&webgpu::GpuShaderModuleDescriptor {
        label: Some("Gemm Shader".to_string()),
        code: include_str!("gemm.wgsl").to_string(),
        compilation_hints: None,
    });

    let pipeline = device.create_compute_pipeline(webgpu::GpuComputePipelineDescriptor {
        label: Some("Gemm Pipeline".to_string()),
        layout: webgpu::GpuLayoutMode::Auto,
        compute: webgpu::GpuProgrammableStage {
            module: &cs_module,
            entry_point: Some("main".to_string()),
            constants: None,
        },
    });

    GpuContext { device, pipeline }
}

fn make_storage_buffer(device: &webgpu::GpuDevice, label: &str, data: &[f32]) -> webgpu::GpuBuffer {
    let bytes = bytemuck::cast_slice(data);
    let buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
        label: Some(label.to_string()),
        size: bytes.len().max(4) as webgpu::GpuSize64,
        usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_DST,
        mapped_at_creation: Some(true),
    });
    buffer.get_mapped_range_set_with_copy(bytes, None, None).unwrap();
    buffer.unmap().unwrap();
    buffer
}

impl Guest for Component {
    fn gemm_load_weights(weight_id: String, weights: Vec<f32>, bias: Vec<f32>) {
        let t_load = std::time::Instant::now(); // [T]

        GPU_CTX.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(build_gpu_context());
            }
            let ctx_ref = cell.borrow();
            let ctx = ctx_ref.as_ref().unwrap();
            let device = &ctx.device;

            let weights_buffer = make_storage_buffer(device, "Weights Buffer (cached)", &weights);
            let bias_buffer = make_storage_buffer(device, "Bias Buffer (cached)", &bias);

            WEIGHTS_CACHE.with(|cache| {
                cache.borrow_mut().insert(
                    weight_id.clone(),
                    CachedWeights { weights_buffer, bias_buffer },
                );
            });
        });

        eprintln!( // [T]
            "[GEMM_LOAD] weight_id={} weights_len={} load_us={}",
            weight_id, weights.len(), t_load.elapsed().as_micros()
        );
    }

    fn gemm(
        weight_id: String,
        in_features: u32,
        out_features: u32,
        mut result_container: Data,
    ) -> Data {
        let result = execute_gemm_gpu(&weight_id, in_features, out_features, &result_container.input);
        result_container.output = result;
        result_container
    }
}

fn execute_gemm_gpu(
    weight_id: &str,
    in_features: u32,
    out_features: u32,
    input: &[f32],
) -> Vec<f32> {
    let t_fn_start = std::time::Instant::now(); // [T]

    GPU_CTX.with(|cell| {
        let ctx_ref = cell.borrow();
        let ctx = ctx_ref.as_ref()
            .expect("GPU_CTX vazio: gemm chamado antes de gemm_load_weights");
        let device = &ctx.device;
        let compute_pipeline = &ctx.pipeline;

        WEIGHTS_CACHE.with(|cache| {
            let cache_ref = cache.borrow();
            let cached = cache_ref.get(weight_id)
                .unwrap_or_else(|| panic!("weight_id '{}' não carregado — chame gemm_load_weights antes", weight_id));

            // Buffer de Parâmetros (Uniform)
            let params: [u32; 2] = [in_features, out_features];
            let params_bytes = bytemuck::cast_slice(&params);
            let params_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
                label: Some("Params Buffer".to_string()),
                size: params_bytes.len() as webgpu::GpuSize64,
                usage: webgpu::GpuBufferUsage::UNIFORM | webgpu::GpuBufferUsage::COPY_DST,
                mapped_at_creation: Some(true),
            });
            params_buffer
                .get_mapped_range_set_with_copy(params_bytes, None, None)
                .unwrap();
            params_buffer.unmap().unwrap();

            let t_input_buf = std::time::Instant::now(); // [T]
            let input_buffer = make_storage_buffer(device, "Input Buffer", input);
            let input_buf_us = t_input_buf.elapsed().as_micros() as u64; // [T]

            // Buffer de Output + Staging
            let output_len = out_features as usize;
            let output_byte_size = (output_len * std::mem::size_of::<f32>()) as webgpu::GpuSize64;
            let output_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
                label: Some("Output Buffer".to_string()),
                size: output_byte_size,
                usage: webgpu::GpuBufferUsage::STORAGE
                    | webgpu::GpuBufferUsage::COPY_SRC
                    | webgpu::GpuBufferUsage::COPY_DST,
                mapped_at_creation: None,
            });

            let staging_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
                label: Some("Staging Buffer".to_string()),
                size: output_byte_size,
                usage: webgpu::GpuBufferUsage::MAP_READ | webgpu::GpuBufferUsage::COPY_DST,
                mapped_at_creation: None,
            });

            // Bind Group — usa os buffers CACHEADOS de weights/bias
            let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&webgpu::GpuBindGroupDescriptor {
                label: Some("Gemm Bind Group".to_string()),
                layout: &bind_group_layout,
                entries: vec![
                    webgpu::GpuBindGroupEntry { binding: 0, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: &params_buffer, offset: Some(0), size: None }) },
                    webgpu::GpuBindGroupEntry { binding: 1, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: &input_buffer, offset: Some(0), size: None }) },
                    webgpu::GpuBindGroupEntry { binding: 2, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: &cached.weights_buffer, offset: Some(0), size: None }) },
                    webgpu::GpuBindGroupEntry { binding: 3, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: &cached.bias_buffer, offset: Some(0), size: None }) },
                    webgpu::GpuBindGroupEntry { binding: 4, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: &output_buffer, offset: Some(0), size: None }) },
                ],
            });

            let t_dispatch = std::time::Instant::now(); // [T]
            let encoder = device.create_command_encoder(Some(&webgpu::GpuCommandEncoderDescriptor { label: None }));
            {
                let cpass = encoder.begin_compute_pass(None);
                cpass.set_pipeline(compute_pipeline);
                cpass
                    .set_bind_group(0, Some(&bind_group), None, None, None)
                    .unwrap();

                let workgroups_x = (out_features + 63) / 64;
                cpass.dispatch_workgroups(workgroups_x, Some(1), Some(1));
                cpass.end();
            }

            encoder.copy_buffer_to_buffer(&output_buffer, None, &staging_buffer, None, None);
            device.queue().submit(&[&encoder.finish(None)]);
            let dispatch_submit_us = t_dispatch.elapsed().as_micros() as u64; // [T]

            let t_readback = std::time::Instant::now(); // [T]
            let data = sync_io::map_and_read_sync(
                &staging_buffer,
                webgpu::GpuMapMode::READ,
                Some(0),
                None,
            )
            .expect("map_and_read_sync falhou no host");
            let readback_us = t_readback.elapsed().as_micros() as u64; // [T]

            let fn_total_us = t_fn_start.elapsed().as_micros() as u64; // [T]
            eprintln!( // [T]
                "[GEMM_TIMING] weight_id={} in_feat={} out_feat={} | input_buf_us={} dispatch_submit_us={} readback_us={} fn_total_us={}",
                weight_id, in_features, out_features,
                input_buf_us, dispatch_submit_us, readback_us, fn_total_us
            );

            if !data.is_empty() {
                bytemuck::cast_slice(&data).to_vec()
            } else {
                vec![0.0f32; output_len]
            }
        })
    })
}

export!(Component);