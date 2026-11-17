//! `edge-codec-cli`: Parse Python or Rust files/stdin into resolved token hints.

use std::io::Read;
use std::path::PathBuf;

use edge_codec::{parse_and_resolve, CodeLanguage};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut lang = CodeLanguage::Python;
    let mut file_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--lang" => {
                i += 1;
                if i < args.len() {
                    lang = match args[i].as_str() {
                        "python" | "py" => CodeLanguage::Python,
                        "rust" | "rs" => CodeLanguage::Rust,
                        other => {
                            eprintln!("unsupported --lang `{other}` (use python|rust)");
                            std::process::exit(2);
                        }
                    };
                }
            }
            "--file" => {
                i += 1;
                if i < args.len() {
                    file_path = Some(PathBuf::from(&args[i]));
                }
            }
            other if !other.starts_with('-') && file_path.is_none() => {
                file_path = Some(PathBuf::from(other));
            }
            other => {
                eprintln!("unknown argument `{other}`");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let source = match file_path {
        Some(p) => std::fs::read_to_string(&p).unwrap_or_else(|e| {
            eprintln!("failed to read {}: {e}", p.display());
            std::process::exit(1);
        }),
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).unwrap();
            buf
        }
    };

    let tokens = parse_and_resolve(&source, lang).unwrap_or_else(|e| {
        eprintln!("parse error: {e}");
        std::process::exit(1);
    });
    println!("{}", serde_json::to_string(&tokens).unwrap());
}
