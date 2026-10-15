wit_bindgen::generate!({
    world: "concatworld",
    path: "wit",
    generate_all,
});

use exports::planner::concatworld::plan::{Data, Guest};

struct Component;

impl Guest for Component {
    fn concat(inputs: Vec<Vec<f32>>, shapes: Vec<Vec<u32>>, axis: u32, result_container: Data) -> Data {
        let output = execute_concat(&inputs, &shapes, axis as usize);

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

/// Concat genérico por eixo. Assume que todos os ramos têm o mesmo rank e
/// os mesmos tamanhos em todas as dims exceto `axis` (garantia do próprio
/// ONNX — senão o grafo original já não validaria). Não há cópia pra GPU
/// aqui: é puro rearranjo de memória (memcpy por fatia), não ganha nada
/// rodando em shader.
fn execute_concat(inputs: &[Vec<f32>], shapes: &[Vec<u32>], axis: usize) -> Vec<f32> {
    assert_eq!(inputs.len(), shapes.len(), "concat: inputs.len() != shapes.len()");
    assert!(!inputs.is_empty(), "concat: nenhum input recebido");

    let rank = shapes[0].len();
    assert!(axis < rank, "concat: axis {} fora do rank {}", axis, rank);

    // outer = produto das dims ANTES do axis; inner = produto das dims
    // DEPOIS do axis. São iguais em todos os ramos — só a dimensão do
    // próprio axis muda de um ramo pro outro.
    let outer: usize = shapes[0][..axis].iter().map(|&d| d as usize).product();
    let inner: usize = shapes[0][axis + 1..].iter().map(|&d| d as usize).product();

    let axis_sizes: Vec<usize> = shapes.iter().map(|s| s[axis] as usize).collect();
    let total_axis: usize = axis_sizes.iter().sum();

    let mut output = vec![0.0f32; outer * total_axis * inner];

    for o in 0..outer {
        let mut running_offset = 0usize;
        for (i, inp) in inputs.iter().enumerate() {
            let axis_size = axis_sizes[i];
            let chunk_len = axis_size * inner;
            let src_start = o * chunk_len;
            let dst_start = o * total_axis * inner + running_offset * inner;

            output[dst_start..dst_start + chunk_len]
                .copy_from_slice(&inp[src_start..src_start + chunk_len]);

            running_offset += axis_size;
        }
    }

    output
}

export!(Component);
