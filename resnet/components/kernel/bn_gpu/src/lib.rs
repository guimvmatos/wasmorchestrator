wit_bindgen::generate!({
    world: "bnworld-gpu",
    path: "wit",
    generate_all,
});

use exports::planner::bnworld::plan::{Data, Guest};
use wasi::webgpu::webgpu;
use wasi_gfx_runtime::wasi_webgpu::sync_io;
use wit_bindgen::block_on;
use std::cell::RefCell;

struct Component;

// Cache de GPU: device + pipeline são idênticos entre chamadas (mesmo
// shader, mesmo bind group layout). Mesmo padrão aplicado no conv_gpu.
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
        "GPU selecionada (BN): vendor='{}' device='{}' architecture='{}'",
        info.vendor(),
        info.device(),
        info.architecture()
    );

    let device = block_on(adapter.request_device(None)).unwrap();

    let cs_module = device.create_shader_module(&webgpu::GpuShaderModuleDescriptor {
        label: Some("BN Shader".to_string()),
        code: include_str!("bn.wgsl").to_string(),
        compilation_hints: None,
    });

    let pipeline = device.create_compute_pipeline(webgpu::GpuComputePipelineDescriptor {
        label: Some("BN Pipeline".to_string()),
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
    fn bn(
        channels: u32,
        h_in: u32,
        w_in: u32,
        epsilon: f32,
        scale: Vec<f32>,
        beta: Vec<f32>,
        mean: Vec<f32>,
        var: Vec<f32>,
        mut img: Data,
    ) -> Data {
        let output_len = img.input.len();

        if img.output.len() != output_len {
            img.output.resize(output_len, 0.0);
        }

        let result = execute_bn_gpu(
            channels, h_in, w_in, epsilon,
            &img.input, &scale, &beta, &mean, &var,
            output_len,
        );

        img.output = result;
        img
    }
}

fn execute_bn_gpu(
    channels: u32,
    h_in: u32,
    w_in: u32,
    epsilon: f32,
    input: &[f32],
    scale: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    output_len: usize,
) -> Vec<f32> {
    GPU_CTX.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(build_gpu_context());
        }
        let ctx_ref = cell.borrow();
        let ctx = ctx_ref.as_ref().unwrap();
        let device = &ctx.device;
        let compute_pipeline = &ctx.pipeline;

        // 3. Buffer de Parâmetros (Uniform)
        let mut params_bytes: Vec<u8> = Vec::new();
        params_bytes.extend_from_slice(bytemuck::cast_slice(&[channels, h_in, w_in]));
        params_bytes.extend_from_slice(bytemuck::cast_slice(&[epsilon]));

        let params_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
            label: Some("Params Buffer".to_string()),
            size: params_bytes.len() as webgpu::GpuSize64,
            usage: webgpu::GpuBufferUsage::UNIFORM | webgpu::GpuBufferUsage::COPY_DST,
            mapped_at_creation: Some(true),
        });
        params_buffer
            .get_mapped_range_set_with_copy(&params_bytes, None, None)
            .unwrap();
        params_buffer.unmap().unwrap();

        // 4. Buffers de entrada (Storage)
        let make_storage_buffer = |label: &str, data: &[f32]| -> webgpu::GpuBuffer {
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
        };

        let input_buffer = make_storage_buffer("Input Buffer", input);
        let scale_buffer = make_storage_buffer("Scale Buffer", scale);
        let beta_buffer   = make_storage_buffer("Beta Buffer", beta);
        let mean_buffer   = make_storage_buffer("Mean Buffer", mean);
        let var_buffer    = make_storage_buffer("Var Buffer", var);

        // 6. Buffer de Output + Staging
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

        // 8. Bind Group
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&webgpu::GpuBindGroupDescriptor {
            label: Some("BN Bind Group".to_string()),
            layout: &bind_group_layout,
            entries: vec![
                webgpu::GpuBindGroupEntry { binding: 0, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: &params_buffer, offset: Some(0), size: None }) },
                webgpu::GpuBindGroupEntry { binding: 1, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: &input_buffer,  offset: Some(0), size: None }) },
                webgpu::GpuBindGroupEntry { binding: 2, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: &scale_buffer,  offset: Some(0), size: None }) },
                webgpu::GpuBindGroupEntry { binding: 3, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: &beta_buffer,   offset: Some(0), size: None }) },
                webgpu::GpuBindGroupEntry { binding: 4, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: &mean_buffer,   offset: Some(0), size: None }) },
                webgpu::GpuBindGroupEntry { binding: 5, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: &var_buffer,    offset: Some(0), size: None }) },
                webgpu::GpuBindGroupEntry { binding: 6, resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding { buffer: &output_buffer, offset: Some(0), size: None }) },
            ],
        });

        // 9. Dispatch
        let encoder = device.create_command_encoder(Some(&webgpu::GpuCommandEncoderDescriptor { label: None }));
        {
            let cpass = encoder.begin_compute_pass(None);
            cpass.set_pipeline(compute_pipeline);
            cpass
                .set_bind_group(0, Some(&bind_group), None, None, None)
                .unwrap();

            let workgroups_x = (w_in + 7) / 8;
            let workgroups_y = (h_in + 7) / 8;
            let workgroups_z = (channels + 3) / 4;

            cpass.dispatch_workgroups(workgroups_x, Some(workgroups_y), Some(workgroups_z));
            cpass.end();
        }

        encoder.copy_buffer_to_buffer(&output_buffer, None, &staging_buffer, None, None);
        device.queue().submit(&[&encoder.finish(None)]);

        // 10. Leitura síncrona
        let data = sync_io::map_and_read_sync(
            &staging_buffer,
            webgpu::GpuMapMode::READ,
            Some(0),
            None,
        )
        .expect("map_and_read_sync falhou no host");

        if !data.is_empty() {
            bytemuck::cast_slice(&data).to_vec()
        } else {
            vec![0.0f32; output_len]
        }
    })
}

export!(Component);