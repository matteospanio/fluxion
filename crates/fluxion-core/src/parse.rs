//! The chain text syntax — one grammar, every interface.
//!
//! This is the exact inverse of [`Graph`]'s [`Display`](std::fmt::Display): whatever the library
//! prints, this parses back to the same graph. That makes the printed form a canonical form, and
//! gives the CLI (`--chain`), Python (`fluxion.chain`), C (`fx_chain_from_text`) and the browser
//! one shared way to describe a chain instead of four.
//!
//! ```
//! use fluxion_core::Graph;
//! let g: Graph = "highpass(80, 4) | gain(-3dB)".parse().unwrap();
//! assert_eq!(g.leaf_count(), 2);
//! assert_eq!(g.to_string().parse::<Graph>().unwrap(), g);
//! ```
//!
//! # Grammar
//!
//! Loosest binding to tightest:
//!
//! ```text
//! chain    = keyed ;
//! keyed    = feedback [ "<" feedback ] ;      (* non-associative *)
//! feedback = series [ "~" series ] ;          (* non-associative *)
//! series   = parallel { "|" parallel } ;      (* left-associative *)
//! parallel = labeled { "+" labeled } ;        (* left-associative *)
//! labeled  = ident ":" labeled | primary ;    (* the label binds tightest *)
//! primary  = "id" | side | op | "(" chain ")" ;
//! side     = "side" "(" digits ")" ;          (* a second input, numbered from 0 *)
//! op       = ident [ "(" [ args ] ")" | "=" values ] ;
//! args     = arg { "," arg } ;
//! arg      = number | ident "=" number ;      (* positional first, then named *)
//! values   = number { "," number } ;
//! number   = [ "-" ] ( digits [ "." digits ] [ exp ] | "inf" ) [ suffix ] ;
//! suffix   = "k" | "dB" ;                     (* case-insensitive *)
//! ```
//!
//! `+` binds tighter than `|`, matching Rust's and Python's operator precedence — so
//! `a | b + c | d` is `a | (b + c) | d` and needs no parentheses.
//!
//! # Side inputs and keys
//!
//! `side(0)` reads the first extra signal handed to the chain instead of what is flowing down it,
//! and `<` says which signal drives a keyed op: `gate(-35, 40) < side(0)` gates the programme by
//! the side signal. The key runs on the same input the node was given, so
//! `gate(-35) < side(0) | lowpass(200)` low-passes the *key*. Only ops that declare a key input
//! read it, so keying a chain of ordinary ops changes nothing.
//!
//! # Filling in parameters
//!
//! Positional arguments fill left to right. Once a named argument appears, every later argument
//! must be named. Anything left unset takes its [`ParamSpec::default`](crate::ParamSpec). So
//! `highpass`, `highpass(80)`, `highpass=80` and `highpass(cutoff=80, order=2)` all mean something
//! sensible. A variadic op (`fir`) takes positional values only, any count ≥ 1.
//!
//! # Suffixes
//!
//! `k` multiplies by 1000 (`1k` = 1000). `dB` converts to the linear factor `10^(x/20)` when the
//! parameter is a plain linear ratio, and is a no-op when the parameter is already in decibels —
//! so `gain(-3dB)` attenuates by 3 dB, which `gain(-3)` emphatically does not. A suffix on a
//! parameter that cannot take it is an error, not a silent misreading. Rendering never emits
//! suffixes, so a round-trip always comes back in the canonical form.

use std::fmt;
use std::str::FromStr;

use crate::graph::Graph;
use crate::op::{Op, OpError, OpKind};
use crate::param::{ParamSpec, Unit};
use crate::suggest;
use crate::tap::TapKind;

/// A syntax or name error in a chain string, located in the source text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// Byte offset into the source string where the problem starts.
    pub offset: usize,
    /// Length in bytes of the offending span (at least 1, so a caret always has something to mark).
    pub len: usize,
    /// What went wrong, e.g. `unknown op 'hipass'`.
    pub message: String,
    /// An optional fix, e.g. `did you mean 'highpass'?`.
    pub help: Option<String>,
}

impl ParseError {
    fn new(offset: usize, len: usize, message: impl Into<String>) -> ParseError {
        ParseError {
            offset,
            len: len.max(1),
            message: message.into(),
            help: None,
        }
    }

    fn with_help(mut self, help: Option<String>) -> ParseError {
        self.help = help;
        self
    }

    /// A three-line rendering with a caret under the offending span — what a terminal should print.
    ///
    /// ```text
    /// error: unknown op 'hipass'
    ///   hipass=80 | gain=-3dB
    ///   ^^^^^^ did you mean 'highpass'?
    /// ```
    pub fn render(&self, src: &str) -> String {
        let at = self.offset.min(src.len());
        let line_start = src[..at].rfind('\n').map_or(0, |i| i + 1);
        let line_end = src[line_start..]
            .find('\n')
            .map_or(src.len(), |i| line_start + i);
        let column = src[line_start..at].chars().count();
        let width = src[at..(at + self.len).min(line_end).max(at)]
            .chars()
            .count()
            .max(1);

        let mut out = format!(
            "error: {}\n  {}\n  {}{}",
            self.message,
            &src[line_start..line_end],
            " ".repeat(column),
            "^".repeat(width),
        );
        if let Some(help) = &self.help {
            out.push(' ');
            out.push_str(help);
        }
        out
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at byte {}: {}", self.offset, self.message)?;
        match &self.help {
            Some(help) => write!(f, " — {help}"),
            None => Ok(()),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse a chain from its text form. See the [module docs](self) for the grammar.
pub fn chain(src: &str) -> Result<Graph, ParseError> {
    let tokens = lex(src)?;
    let mut parser = Parser { tokens, at: 0 };
    let graph = parser.chain()?;
    let rest = parser.peek();
    if rest.tok != Tok::Eof {
        return Err(ParseError::new(
            rest.start,
            rest.text.len(),
            format!("unexpected '{}'", rest.text),
        ));
    }
    Ok(graph)
}

impl FromStr for Graph {
    type Err = ParseError;
    fn from_str(s: &str) -> Result<Graph, ParseError> {
        chain(s)
    }
}

// --- lexer -------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tok {
    Ident,
    Number,
    LParen,
    RParen,
    Pipe,
    Plus,
    Tilde,
    Lt,
    Comma,
    Colon,
    Eq,
    Eof,
}

impl Tok {
    /// How the token is written, for "expected X" messages.
    fn as_str(self) -> &'static str {
        match self {
            Tok::Ident => "a name",
            Tok::Number => "a number",
            Tok::LParen => "(",
            Tok::RParen => ")",
            Tok::Pipe => "|",
            Tok::Plus => "+",
            Tok::Tilde => "~",
            Tok::Lt => "<",
            Tok::Comma => ",",
            Tok::Colon => ":",
            Tok::Eq => "=",
            Tok::Eof => "end of input",
        }
    }
}

#[derive(Clone, Copy)]
struct Spanned<'a> {
    tok: Tok,
    /// The token's source text. For a number this includes any suffix.
    text: &'a str,
    /// Byte offset of the token in the source.
    start: usize,
    /// For a number, how many bytes of `text` are the numeric body (the rest is the suffix).
    /// Zero for every other token.
    body: usize,
}

impl<'a> Spanned<'a> {
    /// The numeric body and the suffix of a number token.
    fn split_number(&self) -> (&'a str, &'a str) {
        self.text.split_at(self.body)
    }
}

fn lex(src: &str) -> Result<Vec<Spanned<'_>>, ParseError> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        let (tok, body) = match c {
            b'(' => (Tok::LParen, 0),
            b')' => (Tok::RParen, 0),
            b'|' => (Tok::Pipe, 0),
            b'+' => (Tok::Plus, 0),
            b'~' => (Tok::Tilde, 0),
            b'<' => (Tok::Lt, 0),
            b',' => (Tok::Comma, 0),
            b':' => (Tok::Colon, 0),
            b'=' => (Tok::Eq, 0),
            // `-` only ever starts a number: the grammar has no subtraction. `+` is always the
            // parallel operator, so a number is never written with a leading `+`.
            b'-' | b'.' | b'0'..=b'9' => {
                let (end, body) = scan_number(bytes, i);
                i = end;
                out.push(Spanned {
                    tok: Tok::Number,
                    text: &src[start..end],
                    start,
                    body: body - start,
                });
                continue;
            }
            b'_' | b'a'..=b'z' | b'A'..=b'Z' => {
                let mut end = i + 1;
                while end < bytes.len()
                    && (bytes[end] == b'_' || bytes[end].is_ascii_alphanumeric())
                {
                    end += 1;
                }
                i = end;
                out.push(Spanned {
                    tok: Tok::Ident,
                    text: &src[start..end],
                    start,
                    body: 0,
                });
                continue;
            }
            _ => {
                // Report the whole character, not the first byte of a multi-byte one.
                let end = src[start..]
                    .char_indices()
                    .nth(1)
                    .map_or(src.len(), |(o, _)| start + o);
                return Err(ParseError::new(
                    start,
                    end - start,
                    format!("unexpected character '{}'", &src[start..end]),
                ));
            }
        };
        i += 1;
        out.push(Spanned {
            tok,
            text: &src[start..i],
            start,
            body,
        });
    }

    out.push(Spanned {
        tok: Tok::Eof,
        text: "",
        start: src.len(),
        body: 0,
    });
    Ok(out)
}

/// Scan a number starting at `i`. Returns `(end_of_token, end_of_numeric_body)` — the bytes in
/// between are the unit suffix.
fn scan_number(bytes: &[u8], i: usize) -> (usize, usize) {
    let mut j = i;
    if bytes[j] == b'-' {
        j += 1;
    }
    if bytes[j..].starts_with(b"inf") {
        j += 3;
    } else {
        while j < bytes.len() && (bytes[j].is_ascii_digit() || bytes[j] == b'.') {
            j += 1;
        }
        // An exponent only counts if it is actually followed by digits, so `1e` lexes as `1` + the
        // suffix `e` (which then fails with a suffix error rather than a confusing number error).
        if j < bytes.len() && (bytes[j] | 0x20) == b'e' {
            let mut k = j + 1;
            if k < bytes.len() && (bytes[k] == b'+' || bytes[k] == b'-') {
                k += 1;
            }
            if k < bytes.len() && bytes[k].is_ascii_digit() {
                j = k;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
            }
        }
    }
    let body_end = j;
    while j < bytes.len() && bytes[j].is_ascii_alphabetic() {
        j += 1;
    }
    (j, body_end)
}

// --- parser ------------------------------------------------------------------------------------

/// One argument as written: its value before unit scaling, plus where it came from.
struct Arg<'a> {
    /// `Some` for `cutoff=800`, `None` for a positional value.
    name: Option<Spanned<'a>>,
    /// The numeric body, before any suffix is applied.
    raw: f32,
    /// The `k` / `dB` suffix, or `""`.
    suffix: &'a str,
    /// The whole argument, for error carets.
    span: Spanned<'a>,
}

struct Parser<'a> {
    tokens: Vec<Spanned<'a>>,
    at: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Spanned<'a> {
        self.tokens[self.at.min(self.tokens.len() - 1)]
    }

    fn peek_next(&self) -> Spanned<'a> {
        self.tokens[(self.at + 1).min(self.tokens.len() - 1)]
    }

    fn bump(&mut self) -> Spanned<'a> {
        let t = self.peek();
        if self.at < self.tokens.len() - 1 {
            self.at += 1;
        }
        t
    }

    fn expect(&mut self, tok: Tok) -> Result<Spanned<'a>, ParseError> {
        let t = self.peek();
        if t.tok == tok {
            return Ok(self.bump());
        }
        Err(ParseError::new(
            t.start,
            t.text.len(),
            format!("expected '{}', found {}", tok.as_str(), found(t)),
        ))
    }

    fn chain(&mut self) -> Result<Graph, ParseError> {
        let node = self.feedback()?;
        if self.peek().tok != Tok::Lt {
            return Ok(node);
        }
        self.bump();
        let key = self.feedback()?;
        // `<` is non-associative for the same reason `~` is: `a < b < c` has no agreed reading.
        let t = self.peek();
        if t.tok == Tok::Lt {
            return Err(
                ParseError::new(t.start, t.text.len(), "'<' cannot be chained").with_help(Some(
                    "bracket the key you mean, e.g. '(a < b) < c'".to_string(),
                )),
            );
        }
        Ok(node.keyed(key))
    }

    fn feedback(&mut self) -> Result<Graph, ParseError> {
        let forward = self.series()?;
        if self.peek().tok != Tok::Tilde {
            return Ok(forward);
        }
        self.bump();
        let feedback = self.series()?;
        // `~` is non-associative: `a ~ b ~ c` has no agreed reading, so say so instead of guessing.
        let t = self.peek();
        if t.tok == Tok::Tilde {
            return Err(
                ParseError::new(t.start, t.text.len(), "'~' cannot be chained").with_help(Some(
                    "bracket the loop you mean, e.g. '(a ~ b) ~ c'".to_string(),
                )),
            );
        }
        Ok(forward.feedback(feedback))
    }

    fn series(&mut self) -> Result<Graph, ParseError> {
        let mut left = self.parallel()?;
        while self.peek().tok == Tok::Pipe {
            self.bump();
            left = left | self.parallel()?;
        }
        Ok(left)
    }

    fn parallel(&mut self) -> Result<Graph, ParseError> {
        let mut left = self.labeled()?;
        while self.peek().tok == Tok::Plus {
            self.bump();
            left = left + self.labeled()?;
        }
        Ok(left)
    }

    fn labeled(&mut self) -> Result<Graph, ParseError> {
        if self.peek().tok == Tok::Ident && self.peek_next().tok == Tok::Colon {
            let name = self.bump();
            self.bump(); // ':'
            // Labels nest: `outer: inner: gain(1)`.
            return Ok(Graph::named(name.text, self.labeled()?));
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<Graph, ParseError> {
        let t = self.peek();
        match t.tok {
            Tok::LParen => {
                self.bump();
                let inner = self.chain()?;
                self.expect(Tok::RParen)?;
                Ok(inner)
            }
            Tok::Ident if t.text == "id" => {
                self.bump();
                Ok(Graph::Id)
            }
            Tok::Ident if t.text == "side" => self.side(),
            Tok::Ident if t.text == "spectrum" || t.text == "meter" => self.tap(),
            Tok::Ident => self.op(),
            _ => Err(ParseError::new(
                t.start,
                t.text.len(),
                format!("expected an op name, found {}", found(t)),
            )),
        }
    }

    /// `side(0)` — read a side signal instead of the chain's own input. Not an op: it takes no
    /// parameters, it takes an *input index*, which is why it is spelled out here next to `id`
    /// rather than being a row in the registry.
    fn side(&mut self) -> Result<Graph, ParseError> {
        let name = self.bump();
        self.expect(Tok::LParen)?;
        let index = self.number()?;
        self.expect(Tok::RParen)?;

        let n = index.raw;
        if !n.is_finite() || n < 0.0 || n.fract() != 0.0 || !index.suffix.is_empty() {
            return Err(ParseError::new(
                index.span.start,
                index.span.text.len(),
                format!(
                    "a side input is numbered 0, 1, 2, … — '{}' is not",
                    index.span.text
                ),
            )
            .with_help(Some(format!(
                "'{}(0)' is the first side signal handed to the chain",
                name.text
            ))));
        }
        Ok(Graph::Side(n as usize))
    }

    /// `spectrum(1024, 0.5)` / `meter` — an observer tap. Not an op for the reason `side` is not
    /// one: it has no effect on the signal, so it is a different kind of node.
    fn tap(&mut self) -> Result<Graph, ParseError> {
        let name = self.bump();
        let mut args = Vec::new();
        if self.peek().tok == Tok::LParen {
            self.bump();
            if self.peek().tok != Tok::RParen {
                loop {
                    args.push(self.number()?);
                    if self.peek().tok != Tok::Comma {
                        break;
                    }
                    self.bump();
                }
            }
            self.expect(Tok::RParen)?;
        }

        let kind = match name.text {
            "meter" => {
                if let Some(extra) = args.first() {
                    return Err(ParseError::new(
                        extra.span.start,
                        extra.span.text.len(),
                        "'meter' takes no parameters",
                    ));
                }
                TapKind::Meter
            }
            _ => {
                if args.len() > 2 {
                    return Err(ParseError::new(
                        args[2].span.start,
                        args[2].span.text.len(),
                        "'spectrum' takes at most 2 parameters: size and overlap",
                    ));
                }
                let size = args.first().map_or(1024.0, |a| a.raw);
                let overlap = args.get(1).map_or(0.5, |a| a.raw);
                if !(size.is_finite() && size >= 2.0) {
                    return Err(ParseError::new(
                        args[0].span.start,
                        args[0].span.text.len(),
                        format!("an FFT size is 2 or more — '{}' is not", args[0].span.text),
                    ));
                }
                if !(0.0..1.0).contains(&overlap) {
                    return Err(ParseError::new(
                        args[1].span.start,
                        args[1].span.text.len(),
                        format!(
                            "overlap is a fraction below 1 — '{}' is not",
                            args[1].span.text
                        ),
                    ));
                }
                TapKind::Spectrum {
                    size: size as usize,
                    overlap,
                }
            }
        };
        Ok(Graph::Tap(kind))
    }

    fn op(&mut self) -> Result<Graph, ParseError> {
        let name = self.bump();
        let kind = OpKind::from_name(name.text).ok_or_else(|| {
            let help = suggest::closest(name.text, OpKind::all().iter().map(|k| k.name()))
                .map(|s| format!("did you mean '{s}'?"));
            ParseError::new(
                name.start,
                name.text.len(),
                format!("unknown op '{}'", name.text),
            )
            .with_help(help)
        })?;

        let mut args = Vec::new();
        match self.peek().tok {
            // `lowpass(800, order=4)`
            Tok::LParen => {
                self.bump();
                if self.peek().tok != Tok::RParen {
                    loop {
                        args.push(self.arg()?);
                        if self.peek().tok != Tok::Comma {
                            break;
                        }
                        self.bump();
                    }
                }
                self.expect(Tok::RParen)?;
            }
            // `highpass=80` / `fir=0.5,0.3`
            Tok::Eq => {
                self.bump();
                loop {
                    args.push(self.number()?);
                    if self.peek().tok != Tok::Comma {
                        break;
                    }
                    self.bump();
                }
            }
            // Bare name: every parameter takes its default.
            _ => {}
        }

        self.build(kind, name, &args).map(Graph::Op)
    }

    fn arg(&mut self) -> Result<Arg<'a>, ParseError> {
        if self.peek().tok == Tok::Ident && self.peek_next().tok == Tok::Eq {
            let name = self.bump();
            self.bump(); // '='
            let mut value = self.number()?;
            value.name = Some(name);
            // The caret should cover `cutoff=800`, not just `800`.
            value.span = Spanned {
                tok: Tok::Number,
                text: name.text,
                start: name.start,
                body: 0,
            };
            return Ok(value);
        }
        self.number()
    }

    fn number(&mut self) -> Result<Arg<'a>, ParseError> {
        let t = self.peek();
        // Rust prints `f32::INFINITY` as `inf`, which lexes as a name; accept it where a number
        // is expected so a graph holding an unbounded parameter still round-trips.
        let (body, suffix) = match t.tok {
            Tok::Number => t.split_number(),
            Tok::Ident if t.text == "inf" => (t.text, ""),
            _ => {
                return Err(ParseError::new(
                    t.start,
                    t.text.len(),
                    format!("expected a number, found {}", found(t)),
                ));
            }
        };
        self.bump();
        let raw = body.parse::<f32>().map_err(|_| {
            ParseError::new(t.start, body.len(), format!("invalid number '{body}'"))
        })?;
        Ok(Arg {
            name: None,
            raw,
            suffix,
            span: t,
        })
    }

    /// Resolve arguments against the op's schema and validate.
    fn build(&self, kind: OpKind, name: Spanned<'a>, args: &[Arg<'a>]) -> Result<Op, ParseError> {
        let specs = kind.params();

        if kind.is_variadic() {
            if let Some(arg) = args.iter().find(|a| a.name.is_some()) {
                return Err(ParseError::new(
                    arg.span.start,
                    arg.span.text.len(),
                    format!(
                        "op '{}' takes a list of values, not named parameters",
                        kind.name()
                    ),
                ));
            }
            let values = if args.is_empty() {
                kind.defaults()
            } else {
                args.iter()
                    .map(|a| scale(a, &specs[0], kind))
                    .collect::<Result<Vec<f32>, _>>()?
            };
            return Op::new(kind, values).map_err(|e| self.op_error(e, kind, name, args));
        }

        let mut values = kind.defaults();
        let mut filled = vec![false; specs.len()];
        let mut seen_named = false;

        for (position, arg) in args.iter().enumerate() {
            let index = match arg.name {
                Some(n) => {
                    seen_named = true;
                    specs.iter().position(|s| s.name == n.text).ok_or_else(|| {
                        let help = suggest::closest(n.text, specs.iter().map(|s| s.name))
                            .map(|s| format!("did you mean '{s}'?"));
                        ParseError::new(
                            n.start,
                            n.text.len(),
                            format!("op '{}' has no parameter '{}'", kind.name(), n.text),
                        )
                        .with_help(help)
                    })?
                }
                None if seen_named => {
                    return Err(ParseError::new(
                        arg.span.start,
                        arg.span.text.len(),
                        "positional value after a named one",
                    )
                    .with_help(Some(format!(
                        "name it too, e.g. '{}=...'",
                        specs.get(position).map_or("param", |s| s.name)
                    ))));
                }
                None if position >= specs.len() => {
                    return Err(ParseError::new(
                        arg.span.start,
                        arg.span.text.len(),
                        format!(
                            "op '{}' takes {} parameter(s), got {}",
                            kind.name(),
                            specs.len(),
                            args.len()
                        ),
                    ));
                }
                None => position,
            };

            if filled[index] {
                return Err(ParseError::new(
                    arg.span.start,
                    arg.span.text.len(),
                    format!(
                        "op '{}' parameter '{}' given twice",
                        kind.name(),
                        specs[index].name
                    ),
                ));
            }
            filled[index] = true;
            values[index] = scale(arg, &specs[index], kind)?;
        }

        Op::new(kind, values).map_err(|e| self.op_error(e, kind, name, args))
    }

    /// Turn an [`OpError`] into a located [`ParseError`], pointing at the argument that caused it
    /// when the user actually wrote one.
    fn op_error(
        &self,
        error: OpError,
        kind: OpKind,
        name: Spanned<'a>,
        args: &[Arg<'a>],
    ) -> ParseError {
        let mut span = name;
        if let OpError::OutOfRange { param, .. } = &error
            && let Some(index) = kind.params().iter().position(|s| s.name == *param)
        {
            // Named argument first, then the positional at that index.
            let culprit = args
                .iter()
                .find(|a| a.name.is_some_and(|n| n.text == *param))
                .or_else(|| args.iter().filter(|a| a.name.is_none()).nth(index));
            if let Some(arg) = culprit {
                span = arg.span;
            }
        }
        ParseError::new(span.start, span.text.len(), error.to_string())
    }
}

/// Apply a `k` / `dB` suffix, given what the target parameter is measured in.
fn scale(arg: &Arg<'_>, spec: &ParamSpec, kind: OpKind) -> Result<f32, ParseError> {
    let at = |message: String| ParseError::new(arg.span.start, arg.span.text.len(), message);

    if arg.suffix.is_empty() {
        return Ok(arg.raw);
    }
    if arg.suffix.eq_ignore_ascii_case("k") {
        return Ok(arg.raw * 1000.0);
    }
    if arg.suffix.eq_ignore_ascii_case("db") {
        return match spec.unit {
            // Already decibels — the suffix just restates the unit.
            Unit::Db => Ok(arg.raw),
            // A plain ratio: this is the conversion that makes `gain(-3dB)` mean −3 dB.
            Unit::Linear => Ok(10f32.powf(arg.raw / 20.0)),
            unit => Err(at(format!(
                "op '{}' parameter '{}' is in {}, so a 'dB' value has no meaning here",
                kind.name(),
                spec.name,
                unit_name(unit),
            ))),
        };
    }
    Err(at(format!("unknown suffix '{}'", arg.suffix))
        .with_help(Some("the suffixes are 'k' (x1000) and 'dB'".to_string())))
}

fn unit_name(unit: Unit) -> &'static str {
    match unit {
        Unit::Hz => "hertz",
        Unit::Db => "decibels",
        Unit::Linear => "linear units",
        Unit::Seconds => "seconds",
        Unit::Q => "Q",
    }
}

/// How to name a token in an error message.
fn found(t: Spanned<'_>) -> String {
    match t.tok {
        Tok::Eof => "end of input".to_string(),
        _ => format!("'{}'", t.text),
    }
}

#[cfg(test)]
mod tests {
    use super::{ParseError, chain};
    use crate::graph::Graph;
    use crate::op::OpKind;

    fn g(src: &str) -> Graph {
        chain(src).unwrap_or_else(|e| panic!("{}", e.render(src)))
    }

    fn err(src: &str) -> ParseError {
        chain(src).expect_err("expected a parse error")
    }

    #[test]
    fn bare_name_uses_every_default() {
        assert_eq!(g("lowpass"), Graph::op(OpKind::Lowpass, [1000.0, 2.0]));
        assert_eq!(g("reverse"), Graph::op(OpKind::Reverse, []));
        assert_eq!(g("id"), Graph::Id);
    }

    #[test]
    fn trailing_parameters_fall_back_to_defaults() {
        assert_eq!(g("highpass(80)"), Graph::op(OpKind::Highpass, [80.0, 2.0]));
        assert_eq!(g("highpass=80"), Graph::op(OpKind::Highpass, [80.0, 2.0]));
        assert_eq!(
            g("highpass(80, 4)"),
            Graph::op(OpKind::Highpass, [80.0, 4.0])
        );
    }

    #[test]
    fn named_parameters_may_follow_positional_ones() {
        assert_eq!(
            g("lowpass(800, order=4)"),
            Graph::op(OpKind::Lowpass, [800.0, 4.0])
        );
        assert_eq!(
            g("lowpass(order=4, cutoff=800)"),
            Graph::op(OpKind::Lowpass, [800.0, 4.0])
        );
        // Six parameters, two worth naming.
        assert_eq!(
            g("compand(threshold=-24, ratio=8)"),
            Graph::op(OpKind::Compand, [0.01, 0.1, -24.0, 8.0, 6.0, 0.0])
        );
    }

    #[test]
    fn suffixes_scale_by_the_parameter_unit() {
        assert_eq!(g("lowpass(1k)"), Graph::op(OpKind::Lowpass, [1000.0, 2.0]));
        assert_eq!(
            g("lowpass(4.41k)"),
            Graph::op(OpKind::Lowpass, [4410.0, 2.0])
        );
        // `gain` is a linear ratio: -3 dB is 0.708, emphatically not -3.
        match g("gain(-3dB)") {
            Graph::Op(op) => assert!((op.params[0] - 0.707_945_8).abs() < 1e-6),
            other => panic!("expected a gain op, got {other}"),
        }
        // `overdrive`'s gain is already in decibels, so the suffix only restates the unit.
        assert_eq!(
            g("overdrive(20dB, 0.2)"),
            Graph::op(OpKind::Overdrive, [20.0, 0.2])
        );
    }

    #[test]
    fn a_suffix_that_makes_no_sense_is_refused() {
        let e = err("lowpass(1dB)");
        assert!(e.message.contains("hertz"), "{}", e.message);
        let e = err("lowpass(1q)");
        assert!(e.message.contains("unknown suffix"), "{}", e.message);
    }

    #[test]
    fn plus_binds_tighter_than_pipe() {
        let a = g("lowpass(200) | highpass(80) + notch(50) | gain(0.5)");
        let b = g("lowpass(200) | (highpass(80) + notch(50)) | gain(0.5)");
        assert_eq!(a, b);
    }

    #[test]
    fn variadic_takes_any_number_of_values() {
        assert_eq!(
            g("fir=0.5,0.3,0.2"),
            Graph::op(OpKind::Fir, [0.5, 0.3, 0.2])
        );
        assert_eq!(g("fir(0.5, 0.3)"), Graph::op(OpKind::Fir, [0.5, 0.3]));
        assert_eq!(g("fir"), Graph::op(OpKind::Fir, [1.0]));
        assert!(err("fir(tap=0.5)").message.contains("not named parameters"));
    }

    #[test]
    fn labels_and_feedback_survive_nesting() {
        assert_eq!(
            g("lp: gain(1) | gain(2)"),
            Graph::named("lp", Graph::op(OpKind::Gain, [1.0])) | Graph::op(OpKind::Gain, [2.0])
        );
        assert_eq!(
            g("mix: (gain(1) | gain(2))"),
            Graph::named(
                "mix",
                Graph::op(OpKind::Gain, [1.0]) | Graph::op(OpKind::Gain, [2.0])
            )
        );
        assert_eq!(
            g("outer: inner: gain(1)"),
            Graph::named(
                "outer",
                Graph::named("inner", Graph::op(OpKind::Gain, [1.0]))
            )
        );
        assert_eq!(
            g("(gain(1) ~ gain(0.5))"),
            Graph::op(OpKind::Gain, [1.0]).feedback(Graph::op(OpKind::Gain, [0.5]))
        );
    }

    #[test]
    fn errors_point_at_the_mistake_and_suggest_a_fix() {
        let e = err("hipass=80 | gain(2)");
        assert_eq!(e.offset, 0);
        assert_eq!(e.message, "unknown op 'hipass'");
        assert_eq!(e.help.as_deref(), Some("did you mean 'highpass'?"));
        assert_eq!(
            e.render("hipass=80 | gain(2)"),
            "error: unknown op 'hipass'\n  hipass=80 | gain(2)\n  ^^^^^^ did you mean 'highpass'?"
        );

        let e = err("lowpass(cutof=800)");
        assert_eq!(e.message, "op 'lowpass' has no parameter 'cutof'");
        assert_eq!(e.help.as_deref(), Some("did you mean 'cutoff'?"));

        // Out-of-range points at the value, not at the op name.
        let e = err("lowpass(-5)");
        assert!(e.message.contains("out of range"), "{}", e.message);
        assert_eq!(e.offset, 8);

        assert!(err("lowpass(800").message.contains("expected ')'"));
        assert!(err("gain(1) |").message.contains("end of input"));
        assert!(err("").message.contains("end of input"));
        assert!(
            err("gain(1) ~ gain(2) ~ gain(3)")
                .message
                .contains("cannot be chained")
        );
        assert!(
            err("lowpass(1, 2, 3)")
                .message
                .contains("takes 2 parameter")
        );
        assert!(
            err("lowpass(cutoff=1, 2)")
                .message
                .contains("positional value after a named one")
        );
        assert!(
            err("lowpass(800, cutoff=900)")
                .message
                .contains("given twice")
        );
        assert!(
            err("gain(1) £")
                .message
                .contains("unexpected character '£'")
        );
    }
}
