wit_bindgen::generate!({
    world: "convworld-gpu",
    path: "wit",
    generate_all,
});

use exports::planner::convworld::plan::{Data, Guest};
use wasi::webgpu::webgpu;
use wasi_gfx_runtime::wasi_webgpu::sync_io;
use wit_bindgen::block_on;

struct Component;

impl Guest for Component {
    fn conv(
        c_out: u32,
        c_in: u32,
        h_in: u32,
        w_in: u32,
        kh: u32,
        kw: u32,
        stride: u32,
        padding: u32,
        weights: Vec<f32>,
        mut img: Data,
    ) -> Data {
        let out_h = ((h_in + 2 * padding - kh) / stride) + 1;
        let out_w = ((w_in + 2 * padding - kw) / stride) + 1;
        let output_len = (c_out * out_h * out_w) as usize;

        if img.output.len() != output_len {
            img.output.resize(output_len, 0.0);
        }

        let result = execute_conv_gpu(
            c_out, c_in, h_in, w_in, kh, kw, stride, padding, out_h, out_w,
            &img.input,
            &weights,
            output_len,
        );

        img.output = result;
        img
    }
}

fn execute_conv_gpu(
    c_out: u32,
    c_in: u32,
    h_in: u32,
    w_in: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    padding: u32,
    out_h: u32,
    out_w: u32,
    input: &[f32],
    weights: &[f32],
    output_len: usize,
) -> Vec<f32> {
    // 1. Conexão síncrona com a GPU
    let gpu = webgpu::get_gpu();
    let adapter_options = webgpu::GpuRequestAdapterOptions {
        feature_level: None,
        power_preference: Some(webgpu::GpuPowerPreference::HighPerformance),
        force_fallback_adapter: None,
        xr_compatible: None,
    };
    let adapter = block_on(gpu.request_adapter(Some(adapter_options))).unwrap();
    // Prova direta de qual GPU está sendo usada — sem depender de monitorar
    // nvidia-smi no timing certo.
    let info = adapter.info();
    eprintln!(
        "GPU selecionada: vendor='{}' device='{}' architecture='{}'",
        info.vendor(),
        info.device(),
        info.architecture()
    );
    let device = block_on(adapter.request_device(None)).unwrap();

    // 2. Criação do Shader Module
    let cs_module = device.create_shader_module(&webgpu::GpuShaderModuleDescriptor {
        label: Some("Conv Shader".to_string()),
        code: include_str!("conv.wgsl").to_string(),
        compilation_hints: None,
    });

    // 3. Buffer de Parâmetros (Uniform)
    let params: [u32; 10] = [
        c_out, c_in, h_in, w_in, kh, kw, stride, padding, out_h, out_w,
    ];
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

    // 4. Buffer de Input (Storage)
    let input_bytes = bytemuck::cast_slice(input);
    let input_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
        label: Some("Input Buffer".to_string()),
        size: input_bytes.len() as webgpu::GpuSize64,
        usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_DST,
        mapped_at_creation: Some(true),
    });
    input_buffer
        .get_mapped_range_set_with_copy(input_bytes, None, None)
        .unwrap();
    input_buffer.unmap().unwrap();

    // 5. Buffer de Weights (Storage)
    let weights_bytes = bytemuck::cast_slice(weights);
    let weights_buffer = device.create_buffer(&webgpu::GpuBufferDescriptor {
        label: Some("Weights Buffer".to_string()),
        size: weights_bytes.len() as webgpu::GpuSize64,
        usage: webgpu::GpuBufferUsage::STORAGE | webgpu::GpuBufferUsage::COPY_DST,
        mapped_at_creation: Some(true),
    });
    weights_buffer
        .get_mapped_range_set_with_copy(weights_bytes, None, None)
        .unwrap();
    weights_buffer.unmap().unwrap();

    // 6. Buffer de Output na GPU e Staging Buffer para leitura na CPU
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

    // 7. Pipeline de Computação
    let compute_pipeline = device.create_compute_pipeline(webgpu::GpuComputePipelineDescriptor {
        label: Some("Conv Pipeline".to_string()),
        layout: webgpu::GpuLayoutMode::Auto,
        compute: webgpu::GpuProgrammableStage {
            module: &cs_module,
            entry_point: Some("main".to_string()),
            constants: None,
        },
    });

    // 8. Bind Group (conecta os 4 buffers aos bindings 0, 1, 2, 3 do WGSL)
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&webgpu::GpuBindGroupDescriptor {
        label: Some("Conv Bind Group".to_string()),
        layout: &bind_group_layout,
        entries: vec![
            webgpu::GpuBindGroupEntry {
                binding: 0,
                resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding {
                    buffer: &params_buffer,
                    offset: Some(0),
                    size: None,
                }),
            },
            webgpu::GpuBindGroupEntry {
                binding: 1,
                resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding {
                    buffer: &input_buffer,
                    offset: Some(0),
                    size: None,
                }),
            },
            webgpu::GpuBindGroupEntry {
                binding: 2,
                resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding {
                    buffer: &weights_buffer,
                    offset: Some(0),
                    size: None,
                }),
            },
            webgpu::GpuBindGroupEntry {
                binding: 3,
                resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding {
                    buffer: &output_buffer,
                    offset: Some(0),
                    size: None,
                }),
            },
        ],
    });

    // 9. Command Encoder & Dispatch
    let encoder = device.create_command_encoder(Some(&webgpu::GpuCommandEncoderDescriptor { label: None }));
    {
        let cpass = encoder.begin_compute_pass(None);
        cpass.set_pipeline(&compute_pipeline);
        cpass
            .set_bind_group(0, Some(&bind_group), None, None, None)
            .unwrap();

        let workgroups_x = (out_w + 7) / 8;
        let workgroups_y = (out_h + 7) / 8;
        let workgroups_z = (c_out + 3) / 4;

        cpass.dispatch_workgroups(workgroups_x, Some(workgroups_y), Some(workgroups_z));
        cpass.end();
    }

    // Copia do output_buffer (VRAM) para staging_buffer
    encoder.copy_buffer_to_buffer(&output_buffer, None, &staging_buffer, None, None);
    device.queue().submit(&[&encoder.finish(None)]);

    // 10. Conclui a leitura do buffer: map + poll bloqueante no host (thread
    // nativa, não task do component model) + cópia + unmap, tudo numa única
    // chamada síncrona. Evita o trap "cannot block a synchronous task
    // before returning" que map_async (import async) causava aqui.
    let data = sync_io::map_and_read_sync(
        &staging_buffer,
        webgpu::GpuMapMode::READ,
        Some(0),
        None,
    )
    .expect("map_and_read_sync falhou no host");

    let result: Vec<f32> = if !data.is_empty() {
        bytemuck::cast_slice(&data).to_vec()
    } else {
        vec![0.0f32; output_len]
    };

    result
}

export!(Component);