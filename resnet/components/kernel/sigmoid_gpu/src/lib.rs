wit_bindgen::generate!({
    world: "sigmoidworld-gpu",
    path: "wit",
    generate_all,
});

use exports::planner::sigmoidworld::plan::{Data, Guest};
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
        "GPU selecionada (Sigmoid): vendor='{}' device='{}' architecture='{}'",
        info.vendor(),
        info.device(),
        info.architecture()
    );

    let device = block_on(adapter.request_device(None)).unwrap();

    let cs_module = device.create_shader_module(&webgpu::GpuShaderModuleDescriptor {
        label: Some("Sigmoid Shader".to_string()),
        code: include_str!("sigmoid.wgsl").to_string(),
        compilation_hints: None,
    });

    let pipeline = device.create_compute_pipeline(webgpu::GpuComputePipelineDescriptor {
        label: Some("Sigmoid Pipeline".to_string()),
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
    fn sigmoid(mut img: Data) -> Data {
        let result = execute_sigmoid_gpu(&img.input);
        img.output = result;
        img
    }
}

fn execute_sigmoid_gpu(input: &[f32]) -> Vec<f32> {
    let t_fn_start = std::time::Instant::now(); // [T]

    GPU_CTX.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(build_gpu_context());
        }
        let ctx_ref = cell.borrow();
        let ctx = ctx_ref.as_ref().unwrap();
        let device = &ctx.device;
        let compute_pipeline = &ctx.pipeline;

        let t_input_buf = std::time::Instant::now(); // [T]
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
        let input_buf_us = t_input_buf.elapsed().as_micros() as u64; // [T]

        let output_byte_size = input_bytes.len() as webgpu::GpuSize64;
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

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&webgpu::GpuBindGroupDescriptor {
            label: Some("Sigmoid Bind Group".to_string()),
            layout: &bind_group_layout,
            entries: vec![
                webgpu::GpuBindGroupEntry {
                    binding: 0,
                    resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding {
                        buffer: &input_buffer,
                        offset: Some(0),
                        size: None,
                    }),
                },
                webgpu::GpuBindGroupEntry {
                    binding: 1,
                    resource: webgpu::GpuBindingResource::GpuBufferBinding(webgpu::GpuBufferBinding {
                        buffer: &output_buffer,
                        offset: Some(0),
                        size: None,
                    }),
                },
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

            let workgroups = ((input.len() as u32) + 63) / 64;
            cpass.dispatch_workgroups(workgroups, Some(1), Some(1));
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
            "[SIGMOID_TIMING] len={} | input_buf_us={} dispatch_submit_us={} readback_us={} fn_total_us={}",
            input.len(), input_buf_us, dispatch_submit_us, readback_us, fn_total_us
        );

        if !data.is_empty() {
            bytemuck::cast_slice(&data).to_vec()
        } else {
            vec![0.0f32; input.len()]
        }
    })
}

export!(Component);