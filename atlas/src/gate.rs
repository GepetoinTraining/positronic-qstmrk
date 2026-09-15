//! Gate 1 for this crate: no runtime source file carries an IEEE type name.
//! Byte-needle construction so the needles never appear whole in this file either.

pub fn source_float_free() {
    let files: [(&str, &str); 17] = [
        ("clock.rs", include_str!("clock.rs")),
        ("cache.rs", include_str!("cache.rs")),
        ("turn.rs", include_str!("turn.rs")),
        ("cub.rs", include_str!("cub.rs")),
        ("ball.rs", include_str!("ball.rs")),
        ("maelstrom.rs", include_str!("maelstrom.rs")),
        ("fiber.rs", include_str!("fiber.rs")),
        ("fiberkernel.rs", include_str!("fiberkernel.rs")),
        ("card.rs", include_str!("card.rs")),
        ("card_ops.rs", include_str!("card_ops.rs")),
        ("card_layer.rs", include_str!("card_layer.rs")),
        ("fold.rs", include_str!("fold.rs")),
        ("cells.rs", include_str!("cells.rs")),
        ("surfaces.rs", include_str!("surfaces.rs")),
        ("matmul.rs", include_str!("matmul.rs")),
        ("forward.rs", include_str!("forward.rs")),
        ("lib.rs", include_str!("lib.rs")),
    ];
    let f = b'f';
    let needles: [[u8; 3]; 2] = [[f, b'3', b'2'], [f, b'6', b'4']];
    for (name, src) in files.iter() {
        for (lno, raw) in src.lines().enumerate() {
            let code = match raw.find("//") {
                Some(p) => &raw[..p],
                None => raw,
            };
            let cb = code.as_bytes();
            for nd in &needles {
                assert!(
                    !cb.windows(3).any(|w| w == &nd[..]),
                    "gate 1: needle in {} line {}: {:?}",
                    name,
                    lno + 1,
                    raw
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn gate_1_source_needles() {
        super::source_float_free();
    }
}

/// Gate 8 for this crate: no runtime source file names the weights' files or their folder.
/// The only reader of the safetensors is the intake; the engine is built from the cells alone.
pub fn source_never_names_the_weights() {
    let files: [(&str, &str); 17] = [
        ("clock.rs", include_str!("clock.rs")),
        ("cache.rs", include_str!("cache.rs")),
        ("turn.rs", include_str!("turn.rs")),
        ("cub.rs", include_str!("cub.rs")),
        ("ball.rs", include_str!("ball.rs")),
        ("maelstrom.rs", include_str!("maelstrom.rs")),
        ("fiber.rs", include_str!("fiber.rs")),
        ("fiberkernel.rs", include_str!("fiberkernel.rs")),
        ("card.rs", include_str!("card.rs")),
        ("card_ops.rs", include_str!("card_ops.rs")),
        ("card_layer.rs", include_str!("card_layer.rs")),
        ("fold.rs", include_str!("fold.rs")),
        ("cells.rs", include_str!("cells.rs")),
        ("surfaces.rs", include_str!("surfaces.rs")),
        ("matmul.rs", include_str!("matmul.rs")),
        ("forward.rs", include_str!("forward.rs")),
        ("lib.rs", include_str!("lib.rs")),
    ];
    let a = String::from("safe") + "tensors";
    let b = String::from("models") + "/";
    for (name, src) in files.iter() {
        for line in src.lines() {
            let code = line.split("//").next().unwrap_or("");
            assert!(!code.contains(&a) && !code.contains(&b), "gate 8: {} names the weights: {}", name, line.trim());
        }
    }
}

#[cfg(test)]
mod gate8 {
    #[test]
    fn gate_8_source_never_names_the_weights() {
        super::source_never_names_the_weights();
    }
}
