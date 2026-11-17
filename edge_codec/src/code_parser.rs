//! Deterministic `tree-sitter` Python & Rust Lexical Scope and Variable-Definition Resolver.
//!
//! Emits per-token `[modality_flag = 1, symbol_id, relative_def_offset, entropy_bucket]` hints
//! matching Milestone 2 of `ANTIGRAVITY.md` and `HintEmbedding` in `schemas/terra_format.md`.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use tree_sitter::{Node, Parser};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum CodeLanguage {
    Python = 0,
    Rust = 1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResolvedToken {
    pub index: usize,
    pub text: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub start_col: usize,
    pub is_identifier: bool,
    pub is_definition: bool,
    pub def_token_index: Option<usize>,
    pub def_line: Option<usize>,
    pub scope_depth: usize,
    /// `[modality_flag = 1, symbol_id, relative_def_offset, entropy_bucket]`
    pub hint: [i32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopeKind {
    Module,
    Function,
    Class,
    Block,
    Comprehension,
}

#[derive(Debug, Clone)]
struct ScopeFrame {
    kind: ScopeKind,
    bindings: HashMap<String, usize>,
    nonlocal_or_global: HashSet<String>,
}

impl ScopeFrame {
    fn new(kind: ScopeKind) -> Self {
        Self {
            kind,
            bindings: HashMap::new(),
            nonlocal_or_global: HashSet::new(),
        }
    }
}

#[inline]
pub fn fnv1a_symbol_id(name: &str) -> i32 {
    let mut hash: u32 = 0x811c_9dc5;
    for &b in name.as_bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    ((hash % 1023) + 1) as i32
}

fn collect_leaf_tokens<'a>(node: Node<'a>, src: &str, out: &mut Vec<ResolvedToken>) {
    if node.child_count() == 0 {
        let sb = node.start_byte();
        let eb = node.end_byte();
        if eb > sb && eb <= src.len() {
            let slice = &src[sb..eb];
            if !slice.trim().is_empty() {
                let idx = out.len();
                let pos = node.start_position();
                out.push(ResolvedToken {
                    index: idx,
                    text: slice.to_string(),
                    start_byte: sb,
                    end_byte: eb,
                    start_line: pos.row,
                    start_col: pos.column,
                    is_identifier: node.kind() == "identifier",
                    is_definition: false,
                    def_token_index: None,
                    def_line: None,
                    scope_depth: 0,
                    hint: [1, 0, 0, 0],
                });
            }
        }
        return;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_leaf_tokens(child, src, out);
    }
}

struct ScopeResolver<'a> {
    src: &'a str,
    lang: CodeLanguage,
    tokens: Vec<ResolvedToken>,
    byte_to_tok: HashMap<usize, usize>,
    scopes: Vec<ScopeFrame>,
}

impl<'a> ScopeResolver<'a> {
    fn new(src: &'a str, lang: CodeLanguage, root: Node<'a>) -> Self {
        let mut tokens = Vec::with_capacity(src.len() / 4);
        collect_leaf_tokens(root, src, &mut tokens);
        let mut byte_to_tok = HashMap::with_capacity(tokens.len());
        for (idx, tok) in tokens.iter().enumerate() {
            byte_to_tok.insert(tok.start_byte, idx);
        }
        Self {
            src,
            lang,
            tokens,
            byte_to_tok,
            scopes: vec![ScopeFrame::new(ScopeKind::Module)],
        }
    }

    #[inline]
    fn scope_depth(&self) -> usize {
        self.scopes.len().saturating_sub(1)
    }

    #[inline]
    fn node_text(&self, node: Node<'_>) -> &'a str {
        &self.src[node.start_byte()..node.end_byte()]
    }

    fn resolve_symbol(&self, name: &str) -> Option<usize> {
        let mut crossed_function_or_comp = false;
        for frame in self.scopes.iter().rev() {
            // In Python, nested functions/methods/comprehensions skip enclosing Class body scopes.
            if self.lang == CodeLanguage::Python
                && crossed_function_or_comp
                && frame.kind == ScopeKind::Class
            {
                continue;
            }
            if let Some(&def_idx) = frame.bindings.get(name) {
                return Some(def_idx);
            }
            if matches!(frame.kind, ScopeKind::Function | ScopeKind::Comprehension) {
                crossed_function_or_comp = true;
            }
        }
        None
    }

    fn mark_definition(&mut self, node: Node<'_>, rebind_in_current_scope: bool) {
        if node.kind() != "identifier" {
            return;
        }
        let Some(&tok_idx) = self.byte_to_tok.get(&node.start_byte()) else {
            return;
        };
        let name = self.node_text(node).to_string();
        let sym_id = fnv1a_symbol_id(&name);
        let depth = self.scope_depth();

        if self.lang == CodeLanguage::Python {
            let is_nonlocal = self
                .scopes
                .last()
                .map(|f| f.nonlocal_or_global.contains(&name))
                .unwrap_or(false);
            if is_nonlocal {
                self.mark_reference(node);
                return;
            }
        }

        let effective_def_idx = {
            let cur = self.scopes.last_mut().expect("at least module scope");
            if rebind_in_current_scope {
                cur.bindings.insert(name, tok_idx);
                tok_idx
            } else {
                *cur.bindings.entry(name).or_insert(tok_idx)
            }
        };

        let def_line = self.tokens[effective_def_idx].start_line;
        let tok = &mut self.tokens[tok_idx];
        tok.is_identifier = true;
        tok.scope_depth = depth;
        if effective_def_idx == tok_idx {
            tok.is_definition = true;
            tok.def_token_index = Some(tok_idx);
            tok.def_line = Some(tok.start_line);
            tok.hint = [1, sym_id, 0, depth.min(3) as i32];
        } else {
            // Re-assignment inside the same Python lexical scope links back to the variable's definition.
            tok.is_definition = false;
            tok.def_token_index = Some(effective_def_idx);
            tok.def_line = Some(def_line);
            tok.hint = [
                1,
                sym_id,
                (effective_def_idx as i32) - (tok_idx as i32),
                depth.min(3) as i32,
            ];
        }
    }

    fn mark_reference(&mut self, node: Node<'_>) {
        if node.kind() != "identifier" {
            return;
        }
        let Some(&tok_idx) = self.byte_to_tok.get(&node.start_byte()) else {
            return;
        };
        let name = self.node_text(node);
        let sym_id = fnv1a_symbol_id(name);
        let depth = self.scope_depth();
        let resolved = self.resolve_symbol(name);
        let def_line = resolved.map(|di| self.tokens[di].start_line);

        let tok = &mut self.tokens[tok_idx];
        tok.is_identifier = true;
        tok.is_definition = false;
        tok.scope_depth = depth;
        tok.def_token_index = resolved;
        tok.def_line = def_line;

        if let Some(def_idx) = resolved {
            let rel_off = (def_idx as i32) - (tok_idx as i32);
            tok.hint = [1, sym_id, rel_off, depth.min(3) as i32];
        } else {
            // Builtin / external symbol: flag with moderate entropy bucket 2.
            tok.hint = [1, sym_id, 0, 2];
        }
    }

    // ------------------------------------------------------------------------
    // Python AST Scope & Binding Traversal
    // ------------------------------------------------------------------------

    fn bind_python_pattern(&mut self, node: Node<'_>, rebind: bool) {
        match node.kind() {
            "identifier" => self.mark_definition(node, rebind),
            "attribute" => {
                // `obj.attr = val`: `obj` is a reference, `attr` is a field name.
                if let Some(obj) = node.child_by_field_name("object") {
                    self.walk_python(obj);
                }
            }
            "subscript" => {
                // `arr[i] = val`: both `arr` and `i` are expressions/references!
                self.walk_python(node);
            }
            "list_splat_pattern"
            | "dictionary_splat_pattern"
            | "tuple_pattern"
            | "list_pattern"
            | "pattern_list"
            | "parenthesized_expression" => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.bind_python_pattern(child, rebind);
                }
            }
            "as_pattern" => {
                // `with ctx() as var` / `except E as e`
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() == "as_pattern_target" {
                        self.bind_python_pattern(child, rebind);
                    } else if child.is_named() {
                        self.walk_python(child);
                    }
                }
            }
            "as_pattern_target" => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.bind_python_pattern(child, rebind);
                }
            }
            _ => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.is_named() {
                        self.bind_python_pattern(child, rebind);
                    }
                }
            }
        }
    }

    fn bind_python_parameters(&mut self, params: Node<'_>, eval_defaults: bool) {
        let mut cursor = params.walk();
        for child in params.children(&mut cursor) {
            match child.kind() {
                "identifier" => {
                    if !eval_defaults {
                        self.mark_definition(child, true);
                    }
                }
                "default_parameter" => {
                    if eval_defaults {
                        if let Some(val) = child.child_by_field_name("value") {
                            self.walk_python(val);
                        }
                    } else if let Some(name) = child.child_by_field_name("name") {
                        self.bind_python_pattern(name, true);
                    }
                }
                "typed_parameter" => {
                    if eval_defaults {
                        if let Some(ty) = child.child_by_field_name("type") {
                            self.walk_python(ty);
                        }
                    } else {
                        let mut c2 = child.walk();
                        for sub in child.children(&mut c2) {
                            if sub.kind() == "type" {
                                continue;
                            }
                            if sub.kind() == "identifier"
                                || sub.kind() == "list_splat_pattern"
                                || sub.kind() == "dictionary_splat_pattern"
                            {
                                self.bind_python_pattern(sub, true);
                                break;
                            }
                        }
                    }
                }
                "typed_default_parameter" => {
                    if eval_defaults {
                        if let Some(ty) = child.child_by_field_name("type") {
                            self.walk_python(ty);
                        }
                        if let Some(val) = child.child_by_field_name("value") {
                            self.walk_python(val);
                        }
                    } else if let Some(name) = child.child_by_field_name("name") {
                        self.bind_python_pattern(name, true);
                    }
                }
                "list_splat_pattern" | "dictionary_splat_pattern" => {
                    if !eval_defaults {
                        self.bind_python_pattern(child, true);
                    }
                }
                _ => {}
            }
        }
    }

    fn walk_python(&mut self, node: Node<'_>) {
        match node.kind() {
            "identifier" => {
                self.mark_reference(node);
            }
            "attribute" => {
                if let Some(obj) = node.child_by_field_name("object") {
                    self.walk_python(obj);
                }
            }
            "keyword_argument" => {
                // `fn(kw=val)`: skip keyword label `name`, resolve `value`
                if let Some(val) = node.child_by_field_name("value") {
                    self.walk_python(val);
                }
            }
            "function_definition" => {
                // Decorators run in enclosing scope before function binding
                let mut dec_cursor = node.walk();
                for child in node.children(&mut dec_cursor) {
                    if child.kind() == "decorator" {
                        self.walk_python(child);
                    }
                }
                if let Some(name) = node.child_by_field_name("name") {
                    self.mark_definition(name, true);
                }
                // Default arg values and parameter annotations evaluate in enclosing scope!
                if let Some(params) = node.child_by_field_name("parameters") {
                    self.bind_python_parameters(params, true);
                }
                if let Some(ret_ty) = node.child_by_field_name("return_type") {
                    self.walk_python(ret_ty);
                }

                self.scopes.push(ScopeFrame::new(ScopeKind::Function));
                if let Some(params) = node.child_by_field_name("parameters") {
                    self.bind_python_parameters(params, false);
                }
                if let Some(body) = node.child_by_field_name("body") {
                    self.walk_python(body);
                }
                self.scopes.pop();
            }
            "lambda" => {
                self.scopes.push(ScopeFrame::new(ScopeKind::Function));
                if let Some(params) = node.child_by_field_name("parameters") {
                    self.bind_python_parameters(params, false);
                }
                if let Some(body) = node.child_by_field_name("body") {
                    self.walk_python(body);
                }
                self.scopes.pop();
            }
            "class_definition" => {
                if let Some(name) = node.child_by_field_name("name") {
                    self.mark_definition(name, true);
                }
                if let Some(bases) = node.child_by_field_name("superclasses") {
                    self.walk_python(bases);
                }
                self.scopes.push(ScopeFrame::new(ScopeKind::Class));
                if let Some(body) = node.child_by_field_name("body") {
                    self.walk_python(body);
                }
                self.scopes.pop();
            }
            "assignment" => {
                // Evaluate RHS before binding LHS targets!
                if let Some(rhs) = node.child_by_field_name("right") {
                    self.walk_python(rhs);
                }
                if let Some(ty) = node.child_by_field_name("type") {
                    self.walk_python(ty);
                }
                if let Some(lhs) = node.child_by_field_name("left") {
                    self.bind_python_pattern(lhs, false);
                }
            }
            "augmented_assignment" => {
                if let Some(rhs) = node.child_by_field_name("right") {
                    self.walk_python(rhs);
                }
                if let Some(lhs) = node.child_by_field_name("left") {
                    self.walk_python(lhs);
                }
            }
            "named_expression" => {
                if let Some(val) = node.child_by_field_name("value") {
                    self.walk_python(val);
                }
                if let Some(name) = node.child_by_field_name("name") {
                    self.bind_python_pattern(name, false);
                }
            }
            "for_statement" => {
                if let Some(rhs) = node.child_by_field_name("right") {
                    self.walk_python(rhs);
                }
                if let Some(lhs) = node.child_by_field_name("left") {
                    self.bind_python_pattern(lhs, false);
                }
                if let Some(body) = node.child_by_field_name("body") {
                    self.walk_python(body);
                }
                if let Some(alt) = node.child_by_field_name("alternative") {
                    self.walk_python(alt);
                }
            }
            "with_item" => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() == "as_pattern" {
                        self.bind_python_pattern(child, false);
                    } else if child.is_named() {
                        self.walk_python(child);
                    }
                }
            }
            "except_clause" => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() == "as_pattern" {
                        self.bind_python_pattern(child, false);
                    } else if child.is_named() {
                        self.walk_python(child);
                    }
                }
            }
            "list_comprehension"
            | "dictionary_comprehension"
            | "set_comprehension"
            | "generator_expression" => {
                let mut clauses = Vec::new();
                let mut body_nodes = Vec::new();
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if !child.is_named() {
                        continue;
                    }
                    if matches!(child.kind(), "for_in_clause" | "if_clause") {
                        clauses.push(child);
                    } else {
                        body_nodes.push(child);
                    }
                }

                // In Python, the outermost `for_in_clause`'s iterable executes in the enclosing scope.
                let mut first_for_right_done = false;
                if let Some(&first_clause) = clauses.first() {
                    if first_clause.kind() == "for_in_clause" {
                        if let Some(right) = first_clause.child_by_field_name("right") {
                            self.walk_python(right);
                            first_for_right_done = true;
                        }
                    }
                }

                self.scopes.push(ScopeFrame::new(ScopeKind::Comprehension));
                for (i, clause) in clauses.into_iter().enumerate() {
                    if clause.kind() == "for_in_clause" {
                        if !(i == 0 && first_for_right_done) {
                            if let Some(right) = clause.child_by_field_name("right") {
                                self.walk_python(right);
                            }
                        }
                        if let Some(left) = clause.child_by_field_name("left") {
                            self.bind_python_pattern(left, true);
                        }
                    } else {
                        self.walk_python(clause);
                    }
                }
                for b in body_nodes {
                    self.walk_python(b);
                }
                self.scopes.pop();
            }
            "global_statement" | "nonlocal_statement" => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() == "identifier" {
                        let name = self.node_text(child).to_string();
                        if let Some(cur) = self.scopes.last_mut() {
                            cur.nonlocal_or_global.insert(name);
                        }
                        self.mark_reference(child);
                    }
                }
            }
            "import_statement" => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    match child.kind() {
                        "aliased_import" => {
                            if let Some(alias) = child.child_by_field_name("alias") {
                                self.mark_definition(alias, false);
                            }
                        }
                        "dotted_name" => {
                            let first_ident = {
                                let mut c2 = child.walk();
                                let found =
                                    child.children(&mut c2).find(|c| c.kind() == "identifier");
                                found
                            };
                            if let Some(fi) = first_ident {
                                self.mark_definition(fi, false);
                            }
                        }
                        "identifier" => self.mark_definition(child, false),
                        _ => {}
                    }
                }
            }
            "import_from_statement" => {
                let mut cursor = node.walk();
                let mut past_import_kw = false;
                for child in node.children(&mut cursor) {
                    if child.kind() == "import" {
                        past_import_kw = true;
                        continue;
                    }
                    if !past_import_kw {
                        continue;
                    }
                    match child.kind() {
                        "aliased_import" => {
                            if let Some(alias) = child.child_by_field_name("alias") {
                                self.mark_definition(alias, false);
                            }
                        }
                        "dotted_name" => {
                            let first_ident = {
                                let mut c2 = child.walk();
                                let found =
                                    child.children(&mut c2).find(|c| c.kind() == "identifier");
                                found
                            };
                            if let Some(fi) = first_ident {
                                self.mark_definition(fi, false);
                            }
                        }
                        "identifier" => self.mark_definition(child, false),
                        _ => {}
                    }
                }
            }
            _ => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.walk_python(child);
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Rust AST Scope & Binding Traversal
    // ------------------------------------------------------------------------

    fn bind_rust_pattern(&mut self, node: Node<'_>) {
        match node.kind() {
            "identifier" => {
                // Skip UPPER_CASE enum/const path segments? Local pattern idents bind!
                self.mark_definition(node, true);
            }
            "scoped_identifier" | "type_identifier" | "field_identifier" => {}
            "field_pattern" => {
                if let Some(pat) = node.child_by_field_name("pattern") {
                    self.bind_rust_pattern(pat);
                }
            }
            _ => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.bind_rust_pattern(child);
                }
            }
        }
    }

    fn walk_rust(&mut self, node: Node<'_>) {
        match node.kind() {
            "identifier" => self.mark_reference(node),
            "field_identifier" | "type_identifier" | "primitive_type" => {}
            "field_expression" => {
                if let Some(val) = node.child_by_field_name("value") {
                    self.walk_rust(val);
                }
            }
            "function_item" => {
                if let Some(name) = node.child_by_field_name("name") {
                    self.mark_definition(name, true);
                }
                self.scopes.push(ScopeFrame::new(ScopeKind::Function));
                if let Some(params) = node.child_by_field_name("parameters") {
                    let mut c = params.walk();
                    for param in params.children(&mut c) {
                        if param.kind() == "parameter" {
                            if let Some(pat) = param.child_by_field_name("pattern") {
                                self.bind_rust_pattern(pat);
                            }
                        } else if param.kind() == "self_parameter" {
                            let mut sc = param.walk();
                            for sub in param.children(&mut sc) {
                                if sub.kind() == "self" {
                                    self.mark_definition(sub, true);
                                }
                            }
                        }
                    }
                }
                if let Some(body) = node.child_by_field_name("body") {
                    self.walk_rust(body);
                }
                self.scopes.pop();
            }
            "closure_expression" => {
                self.scopes.push(ScopeFrame::new(ScopeKind::Function));
                if let Some(params) = node.child_by_field_name("parameters") {
                    self.bind_rust_pattern(params);
                }
                if let Some(body) = node.child_by_field_name("body") {
                    self.walk_rust(body);
                }
                self.scopes.pop();
            }
            "let_declaration" => {
                if let Some(val) = node.child_by_field_name("value") {
                    self.walk_rust(val);
                }
                if let Some(pat) = node.child_by_field_name("pattern") {
                    self.bind_rust_pattern(pat);
                }
            }
            "for_expression" => {
                if let Some(val) = node.child_by_field_name("value") {
                    self.walk_rust(val);
                }
                self.scopes.push(ScopeFrame::new(ScopeKind::Block));
                if let Some(pat) = node.child_by_field_name("pattern") {
                    self.bind_rust_pattern(pat);
                }
                if let Some(body) = node.child_by_field_name("body") {
                    self.walk_rust(body);
                }
                self.scopes.pop();
            }
            "block" => {
                self.scopes.push(ScopeFrame::new(ScopeKind::Block));
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.walk_rust(child);
                }
                self.scopes.pop();
            }
            "match_arm" => {
                self.scopes.push(ScopeFrame::new(ScopeKind::Block));
                if let Some(pat) = node.child_by_field_name("pattern") {
                    self.bind_rust_pattern(pat);
                }
                if let Some(val) = node.child_by_field_name("value") {
                    self.walk_rust(val);
                }
                self.scopes.pop();
            }
            _ => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.walk_rust(child);
                }
            }
        }
    }

    fn run(mut self, root: Node<'a>) -> Vec<ResolvedToken> {
        match self.lang {
            CodeLanguage::Python => self.walk_python(root),
            CodeLanguage::Rust => self.walk_rust(root),
        }
        self.tokens
    }
}

pub fn parse_and_resolve(source: &str, lang: CodeLanguage) -> Result<Vec<ResolvedToken>, String> {
    let mut parser = Parser::new();
    let ts_lang: tree_sitter::Language = match lang {
        CodeLanguage::Python => tree_sitter_python::LANGUAGE.into(),
        CodeLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
    };
    parser
        .set_language(&ts_lang)
        .map_err(|e| format!("failed to configure tree-sitter language: {e}"))?;

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| "tree-sitter returned None parse tree".to_string())?;
    let resolver = ScopeResolver::new(source, lang, tree.root_node());
    Ok(resolver.run(tree.root_node()))
}

pub fn parse_hint_matrix(source: &str, lang: CodeLanguage) -> Result<Vec<[i32; 4]>, String> {
    Ok(parse_and_resolve(source, lang)?
        .into_iter()
        .map(|t| t.hint)
        .collect())
}
