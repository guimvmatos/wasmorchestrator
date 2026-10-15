wit_bindgen::generate!({
    world: "splitworld",
    path: "wit",
    generate_all,
});

use exports::planner::splitworld::plan::{Data, Guest};

struct Component;

impl Guest for Component {
    fn split(
        input: Vec<f32>,
        shape: Vec<u32>,
        axis: u32,
        sizes: Vec<u32>,
        _result_container: Data,
    ) -> Vec<Vec<f32>> {
        execute_split(&input, &shape, axis as usize, &sizes)
    }
}

/// Inverso do Concat: fatia `input` em `sizes.len()` pedaços ao longo de
/// `axis`. Puro rearranjo de memória, sem ganho em GPU.
fn execute_split(input: &[f32], shape: &[u32], axis: usize, sizes: &[u32]) -> Vec<Vec<f32>> {
    assert!(!sizes.is_empty(), "split: nenhum tamanho de saída recebido");
    let rank = shape.len();
    assert!(axis < rank, "split: axis {} fora do rank {}", axis, rank);

    let outer: usize = shape[..axis].iter().map(|&d| d as usize).product();
    let inner: usize = shape[axis + 1..].iter().map(|&d| d as usize).product();
    let total_axis: usize = sizes.iter().map(|&s| s as usize).sum();

    assert_eq!(
        total_axis, shape[axis] as usize,
        "split: soma dos tamanhos ({}) != shape[axis] ({})",
        total_axis, shape[axis]
    );

    let mut outputs: Vec<Vec<f32>> = sizes
        .iter()
        .map(|&s| vec![0.0f32; outer * (s as usize) * inner])
        .collect();

    let mut offset = 0usize; // posição corrente no axis, em elementos (não floats)
    for (i, &size) in sizes.iter().enumerate() {
        let size = size as usize;
        let chunk_len = size * inner;
        for o in 0..outer {
            let src_start = o * total_axis * inner + offset * inner;
            let dst_start = o * chunk_len;
            outputs[i][dst_start..dst_start + chunk_len]
                .copy_from_slice(&input[src_start..src_start + chunk_len]);
        }
        offset += size;
    }

    outputs
}

export!(Component);
