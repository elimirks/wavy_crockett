use std::{fmt::Debug, cell::RefCell, rc::Rc, collections::{HashMap, HashSet}, process::exit, f64::consts::PI};
use rand::{thread_rng, Rng};
use splines::{Interpolation, Key, Spline};

use crate::parser::*;
use crate::sound_handler::*;
use crate::math::are_points_sorted;

type RunResult<T> = Result<T, String>;

#[derive(Debug)]
struct Scope {
    values: HashMap<String, Rc<Value>>,
    parent: Option<Rc<RefCell<Scope>>>,
}

impl Scope {
    fn new(parent: Option<Rc<RefCell<Scope>>>) -> Scope {
        Scope {
            values: HashMap::new(),
            parent,
        }
    }

    fn lookup(&self, name: &str) -> Rc<Value> {
        if let Some(value) = self.values.get(name) {
            value.clone()
        } else if let Some(parent) = &self.parent {
            parent.borrow().lookup(name)
        } else {
            Value::nil()
        }
    }

    fn insert(&mut self, name: &str, value: Rc<Value>) {
        self.values.insert(name.to_string(), value);
    }
}

struct RunContext {
    scope: Rc<RefCell<Scope>>,
    required_paths: HashSet<String>,
}

impl RunContext {
    fn new() -> Self {
        RunContext {
            scope: Rc::new(RefCell::new(Scope::new(None))),
            required_paths: HashSet::new(),
        }
    }

    fn root_scope(&self) -> Rc<RefCell<Scope>> {
        let mut root = self.scope.clone();
        while root.borrow().parent.is_some() {
            let parent = root.borrow().parent.clone().unwrap();
            root = parent;
        }
        root
    }
}

impl Debug for RunContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RunContext:\n")?;
        let mut frame_index = 0;
        let mut current = Some(self.scope.clone());
        while let Some(scope) = current {
            f.write_str(&format!("Stack frame {frame_index}:\n"))?;
            for (name, value) in scope.borrow().values.iter() {
                f.write_str(&format!("{name} -> {value:?}\n"))?;
            }
            current = scope.borrow().parent.clone();
            frame_index += 1;
        }
        Ok(())
    }
}

pub fn run(file_path: &str) -> RunResult<()> {
    run_with_context(&mut RunContext::new(), file_path)
}

fn run_with_context(ctx: &mut RunContext, file_path: &str) -> RunResult<()> {
    let cwd = std::env::current_dir()
        .expect("Can't find cwd!");
    let path = format!("{}/{file_path}.lisp", cwd.to_str().unwrap());
    if ctx.required_paths.contains(&path) {
        return Ok(());
    }
    ctx.required_paths.insert(path.clone());
    if let Ok(content) = std::fs::read_to_string(&path) {
        let root_exprs = parse_str(&content)?;
        eval_progn(ctx, &root_exprs)?;
        Ok(())
    } else {
        Err(format!("Failed reading {path}"))
    }
}

fn eval(ctx: &mut RunContext, expr: Rc<SExpr>) -> RunResult<Rc<Value>> {
    match &*expr {
        SExpr::Atom(value) => {
            match &**value {
                Value::Symbol(name) => {
                    Ok(ctx.scope.borrow().lookup(name))
                },
                _ => Ok(value.clone()),
            }
        },
        SExpr::S(car, cdr) => {
            let callee = eval(ctx, car.clone())?;
            if let Value::Builtin(Builtin::Quote) = &*callee {
                return Ok(call_quote(cdr.clone()));
            }
            call(ctx, callee, &unfold(cdr.clone()))
        },
    }
}

fn call_quote(arg: Rc<SExpr>) -> Rc<Value> {
    match &*arg {
        SExpr::Atom(value) => value.clone(),
        SExpr::S(car, cdr) => {
            Rc::new(Value::Cons(call_quote(car.clone()), call_quote(cdr.clone())))
        },
    }
}

// The result is the last evaluated expr
fn eval_progn(ctx: &mut RunContext, exprs: &[Rc<SExpr>]) -> RunResult<Rc<Value>> {
    match eval_all(ctx, exprs)?.last() {
        Some(value) => Ok(value.clone()),
        None => Ok(Value::nil())
    }
}

fn eval_all(ctx: &mut RunContext, exprs: &[Rc<SExpr>]) -> RunResult<Vec<Rc<Value>>> {
    exprs.iter().map(|expr| eval(ctx, expr.clone())).collect::<RunResult<Vec<_>>>()
}

fn call(ctx: &mut RunContext, func: Rc<Value>, params: &[Rc<SExpr>]) -> RunResult<Rc<Value>> {
    match &*func {
        Value::Builtin(bi) => call_builtin(ctx, *bi, params),
        Value::Function(captured_scope, args, body) => {
            if params.len() != args.len() {
                return Err("Function call parameter length mismatch".to_owned());
            }
            let param_values = eval_all(ctx, params)?;
            let parent_scope = ctx.scope.clone();
            let mut fun_scope = Scope::new(Some(parent_scope.clone()));
            for (name, value) in captured_scope.iter() {
                match &**value {
                    Value::Builtin(Builtin::Nil) => {},
                    _ => fun_scope.insert(name, value.clone()),
                }
            }
            for (name, param) in args.iter().zip(param_values.iter()) {
                fun_scope.insert(name, param.clone());
            }
            ctx.scope = Rc::new(RefCell::new(fun_scope));
            let result = eval_progn(ctx, body)?;
            ctx.scope = parent_scope;
            Ok(result)
        },
        value => Err(format!("{value:?} is not callable")),
    }
}

/// Unfolds an sexpr list into a Vec
/// Unfolding will terminate when it hits an atom value in the rhs
/// If a nil is the terminal element, it isn't included in the return vec
fn unfold(sexpr: Rc<SExpr>) -> Vec<Rc<SExpr>> {
    let mut values = vec![];
    let mut current = sexpr;
    while let SExpr::S(lhs, rhs) = &*current {
        values.push(lhs.clone());
        current = rhs.clone();
    }
    if !current.is_nil() {
        values.push(current);
    }
    values
}

fn param_count_eq<T>(func: Builtin, params: &[T], n: usize) -> RunResult<()> {
    if params.len() != n {
        Err(format!("{func:?} must have exactly {n} params"))
    } else {
        Ok(())
    }
}

fn param_count_ge<T>(func: Builtin, params: &[T], n: usize) -> RunResult<()> {
    if params.len() < n {
        Err(format!("{func:?} must have at least {n} params"))
    } else {
        Ok(())
    }
}

fn call_builtin(ctx: &mut RunContext, func: Builtin, params: &[Rc<SExpr>]) -> RunResult<Rc<Value>> {
    match func {
        Builtin::Lambda => eval_lambda(ctx, params),
        Builtin::Defun => {
            param_count_ge(func, params, 3)?;
            if let Some(name) = get_expr_symbol_name(&params[0]) {
                let fun_sym = Rc::new(Value::Symbol(name));
                let new = params[1..].to_vec();
                let fun = eval_lambda(ctx, &new)?;
                eval_set(ctx.root_scope(), &[fun_sym, fun])
            } else {
                Err("The first param to defun be a symbol".to_owned())
            }
        },
        Builtin::Nil => Ok(Value::nil()),
        Builtin::Set => {
            let values = eval_all(ctx, params)?;
            eval_set(ctx.scope.clone(), &values)
        },
        Builtin::Setg => {
            let values = eval_all(ctx, params)?;
            eval_set(ctx.root_scope(), &values)
        },
        Builtin::Progn => eval_progn(ctx, params),
        Builtin::Putc => {
            param_count_eq(func, params, 1)?;
            let param = eval(ctx, params[0].clone())?;
            if let Some(c) = try_get_char(&param) {
                print!("{c}");
                Ok(Value::nil())
            } else {
                Err("putc must accept exactly 1 char argument".to_owned())
            }
        },
        Builtin::Debug => {
            param_count_eq(func, params, 1)?;
            let param = params[0].clone(); 
            eprintln!("DEBUG: {param:?}");
            Ok(Value::nil())
        },
        Builtin::If => {
            param_count_eq(func, params, 3)?;
            let cond = eval(ctx, params[0].clone())?;
            if is_truthy(cond) {
                eval(ctx, params[1].clone())
            } else {
                eval(ctx, params[2].clone())
            }
        },
        Builtin::Car => {
            param_count_eq(func, params, 1)?;
            let param = eval(ctx, params[0].clone())?;
            if let Some(car) = get_car(&param) {
                Ok(car)
            } else {
                Err("The argument to car must be a list".to_owned())
            }
        },
        Builtin::Cdr => {
            param_count_eq(func, params, 1)?;
            let param = eval(ctx, params[0].clone())?;
            if let Some(car) = get_cdr(&param) {
                Ok(car)
            } else {
                Err("The argument to cdr must be a list".to_owned())
            }
        },
        Builtin::Cons => {
            param_count_eq(func, params, 2)?;
            let car = eval(ctx, params[0].clone())?;
            let cdr = eval(ctx, params[1].clone())?;
            Ok(Rc::new(Value::Cons(car, cdr)))
        },
        Builtin::IsFalsy => {
            param_count_eq(func, params, 1)?;
            let param = eval(ctx, params[0].clone())?;
            if is_truthy(param) {
                Ok(Value::nil())
            } else {
                Ok(Value::truthy())
            }
        },
        Builtin::IsEq => {
            param_count_eq(func, params, 2)?;
            let lhs = eval(ctx, params[0].clone())?;
            let rhs = eval(ctx, params[1].clone())?;
            if lhs == rhs {
                Ok(Value::truthy())
            } else {
                Ok(Value::nil())
            }
        },
        Builtin::Exit => {
            param_count_eq(func, params, 1)?;
            let param = eval(ctx, params[0].clone())?;
            if let Some(status) = try_get_int(&param).filter(|v| *v >= 0 && *v <= 255) {
                exit(status as i32);
            } else {
                Err(format!("{func:?} must be called on an int value between 0-255"))
            }
        },
        Builtin::Require => {
            param_count_eq(func, params, 1)?;
            let param = eval(ctx, params[0].clone())?;
            if let Some(file_path) = try_get_string(&param) {
                run_with_context(ctx, &file_path)?;
                Ok(Value::nil())
            } else {
                Err(format!("{func:?} accepts exactly 1 string argument"))
            }
        },
        Builtin::Add | Builtin::Sub | Builtin::Mul | Builtin::Div | Builtin::Mod | Builtin::Pow => {
            param_count_eq(func, params, 2)?;
            let lhs = eval(ctx, params[0].clone())?;
            let rhs = eval(ctx, params[1].clone())?;
            eval_arithmetic(func, &lhs, &rhs)
        },
        Builtin::WdPureTone => {
            param_count_eq(func, params, 2)?;
            let values = eval_all(ctx, params)?;
            wd_pure_tone(ctx, &values)
        },
        Builtin::WdSquare => {
            param_count_eq(func, params, 2)?;
            let values = eval_all(ctx, params)?;
            wd_square(ctx, &values)
        },
        Builtin::WdSaw => {
            param_count_eq(func, params, 2)?;
            let values = eval_all(ctx, params)?;
            wd_saw(ctx, &values)
        },
        Builtin::WdTriangle => {
            param_count_eq(func, params, 2)?;
            let values = eval_all(ctx, params)?;
            wd_triangle(ctx, &values)
        },
        Builtin::WdSave => {
            param_count_eq(func, params, 2)?;
            let values = eval_all(ctx, params)?;
            wd_save(ctx, &values)
        },
        Builtin::WdPlay => {
            param_count_eq(func, params, 1)?;
            let values = eval_all(ctx, params)?;
            wd_play(ctx, &values)
        },
        Builtin::WdMultiply => {
            param_count_eq(func, params, 2)?;
            let values = eval_all(ctx, params)?;
            wd_multiply(&values)
        },
        Builtin::WdSuperimpose => {
            param_count_eq(func, params, 2)?;
            let values = eval_all(ctx, params)?;
            wd_superimpose(&values)
        },
        Builtin::WdSuperimposeInsert => {
            param_count_eq(func, params, 3)?;
            let values = eval_all(ctx, params)?;
            wd_superimpose_insert(&values)
        },
        Builtin::WdLen => {
            param_count_eq(func, params, 1)?;
            let values = eval_all(ctx, params)?;
            wd_len(&values)
        },
        Builtin::WdConcat => {
            param_count_eq(func, params, 2)?;
            let values = eval_all(ctx, params)?;
            wd_concat(&values)
        },
        Builtin::WdNoise => {
            param_count_eq(func, params, 1)?;
            let values = eval_all(ctx, params)?;
            wd_noise(&values)
        },
        Builtin::WdSubSample => {
            param_count_eq(func, params, 3)?;
            let values = eval_all(ctx, params)?;
            wd_subsample(&values)
        },
        Builtin::WdReverse => {
            param_count_eq(func, params, 1)?;
            let values = eval_all(ctx, params)?;
            wd_reverse(&values)
        },
        Builtin::WdPlot => {
            param_count_eq(func, params, 1)?;
            let values = eval_all(ctx, params)?;
            wd_plot(&values)
        },
        Builtin::WdFromFrequencies => {
            param_count_eq(func, params, 1)?;
            let values = eval_all(ctx, params)?;
            wd_from_frequencies(ctx, &values)
        },
        Builtin::WdSpline => {
            param_count_eq(func, params, 2)?;
            let values = eval_all(ctx, params)?;
            wd_spline(&values)
        },
        Builtin::WdLowPass => {
            param_count_eq(func, params, 2)?;
            let values = eval_all(ctx, params)?;
            wd_low_pass(ctx, &values)
        },
        Builtin::WdGaussianBlur => {
            param_count_eq(func, params, 2)?;
            let values = eval_all(ctx, params)?;
            wd_gaussian_blur(&values)
        },
        Builtin::ToString => {
            param_count_eq(func, params, 1)?;
            let value = eval(ctx, params[0].clone())?;
            Ok(sexpr_as_string(&value))
        },
        // Special case, handled elsewhere
        Builtin::Quote => unreachable!(),
        Builtin::StrAsList => {
            param_count_eq(func, params, 1)?;
            let param = eval(ctx, params[0].clone())?;
            string_as_char_list(&param)
        },
        Builtin::ListAsStr => {
            param_count_eq(func, params, 1)?;
            let param = eval(ctx, params[0].clone())?;
            char_list_as_string(param)
        },
        Builtin::List => {
            let param_values = eval_all(ctx, params)?;
            Ok(param_values.iter().rev().fold(Value::nil(), |acc, it| {
                Rc::new(Value::Cons(it.clone(), acc))
            }))
        },
        Builtin::Cmp => {
            param_count_eq(func, params, 2)?;
            let lhs = eval(ctx, params[0].clone())?;
            let rhs = eval(ctx, params[1].clone())?;
            let res = match (&*lhs, &*rhs) {
                (Value::Int(lhs), Value::Int(rhs)) => {
                    match lhs.cmp(rhs) {
                        std::cmp::Ordering::Less => -1,
                        std::cmp::Ordering::Equal => 0,
                        std::cmp::Ordering::Greater => 1,
                    }
                },
                (Value::String(lhs), Value::String(rhs)) => {
                    match lhs.cmp(rhs) {
                        std::cmp::Ordering::Less => -1,
                        std::cmp::Ordering::Equal => 0,
                        std::cmp::Ordering::Greater => 1,
                    }
                },
                (Value::Float(lhs), Value::Float(rhs)) => {
                    match lhs.partial_cmp(rhs) {
                        Some(std::cmp::Ordering::Less) => -1,
                        Some(std::cmp::Ordering::Equal) => 0,
                        Some(std::cmp::Ordering::Greater) => 1,
                        None => return Err(format!("Cannot compare {lhs:?} with {rhs:?}")),
                    }
                },
                _ => return Err(format!("Cannot compare {:?} with {:?}", params[0], params[1])),
            };
            Ok(Rc::new(Value::Int(res)))
        },
        Builtin::ToInt => {
            param_count_eq(func, params, 1)?;
            let value = eval(ctx, params[0].clone())?;
            let int_value = match &*value {
                Value::Int(value) => *value,
                Value::Float(value) => *value as i64,
                _ => return Err("to-int only works on floats (and ints)".to_owned()),
            };
            Ok(Rc::new(Value::Int(int_value)))
        },
    }
}

fn wd_pure_tone(ctx: &RunContext, params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let sr = ctx.scope.borrow().lookup("wd-sample-rate");
    let sample_rate = try_get_int(&sr)
        .ok_or("wd-sample-rate must be globally set as an int")?;
    let frequency = try_get_float(&params[0])
        .ok_or("frequency parameter must be a float")?;
    let sample_count = try_get_int(&params[1])
        .ok_or("sample-count parameter must be an int")?;

    let mut data = vec![];
    for t in 0..sample_count as usize {
        let sample_time = (t as f64) / (sample_rate as f64);
        let x = 2.0 * PI * sample_time * frequency;
        data.push(x.sin());
    }
    Ok(Rc::new(Value::WaveData(data)))
}

fn wd_square(ctx: &RunContext, params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let sr = ctx.scope.borrow().lookup("wd-sample-rate");
    let sample_rate = try_get_int(&sr)
        .ok_or("wd-sample-rate must be globally set as an int")?;
    let frequency = try_get_float(&params[0])
        .ok_or("frequency parameter must be a float")?;
    let sample_count = try_get_int(&params[1])
        .ok_or("sample-count parameter must be an int")?;

    let period = 1.0 / frequency;
    let mut data = vec![];
    for t in 0..sample_count as usize {
        let sample_time = (t as f64) / (sample_rate as f64);
        if sample_time % period < period / 2.0 {
            data.push(1.0);
        } else {
            data.push(-1.0);
        }
    }
    Ok(Rc::new(Value::WaveData(data)))
}

fn wd_saw(ctx: &RunContext, params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let sr = ctx.scope.borrow().lookup("wd-sample-rate");
    let sample_rate = try_get_int(&sr)
        .ok_or("wd-sample-rate must be globally set as an int")?;
    let frequency = try_get_float(&params[0])
        .ok_or("frequency parameter must be a float")?;
    let sample_count = try_get_int(&params[1])
        .ok_or("sample-count parameter must be an int")?;

    let period = 1.0 / frequency;
    let mut data = vec![];
    for t in 0..sample_count as usize {
        let t = (t as f64) / (sample_rate as f64);
        let x = 2.0 * ((t / period) - ((t / period) + 0.5).floor());
        data.push(x);
    }
    Ok(Rc::new(Value::WaveData(data)))
}

fn wd_triangle(ctx: &RunContext, params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let sr = ctx.scope.borrow().lookup("wd-sample-rate");
    let sample_rate = try_get_int(&sr)
        .ok_or("wd-sample-rate must be globally set as an int")?;
    let frequency = try_get_float(&params[0])
        .ok_or("frequency parameter must be a float")?;
    let sample_count = try_get_int(&params[1])
        .ok_or("sample-count parameter must be an int")?;

    let period = 1.0 / frequency;
    let mut data = vec![];
    for t in 0..sample_count as usize {
        let t = period / 4.0 + (t as f64) / (sample_rate as f64);
        let x = 2.0 * (t / period - (t / period + 0.5).floor()).abs();
        data.push((2.0 * x) - 1.0);
    }
    Ok(Rc::new(Value::WaveData(data)))
}


fn wd_plot(params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let wavedata = try_get_wavedata(&params[0])
        .ok_or("wavedata parameter must be a wavedata object")?;
    plot_wavedata(wavedata);
    Ok(params[0].clone())
}

// https://math.stackexchange.com/questions/1820065/equation-for-a-sinusoidal-wave-with-changing-frequency
fn wd_from_frequencies(ctx: &RunContext, params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let sr = ctx.scope.borrow().lookup("wd-sample-rate");
    let sample_rate = try_get_int(&sr)
        .ok_or("wd-sample-rate must be globally set as an int")?;
    let frequencies = try_get_wavedata(&params[0])
        .ok_or("frequency-series parameter must be a wavedata object")?;
    if frequencies.iter().any(|f| *f < 0.0) {
        return Err("All frequencies must be equal or above 0.0".to_owned());
    }
    let step_size = 1.0 / (sample_rate as f64);
    let mut cumulative = 0.0;
    let mut data = vec![];
    for f in frequencies.iter() {
        // s(t) = sin(2 pi integral_0^t f(t) dt)
        cumulative += f * step_size;
        let x = 2.0 * PI * cumulative;
        data.push(x.sin());
    }
    Ok(Rc::new(Value::WaveData(data)))
}

fn wd_spline(params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let points = try_get_point_list(params[0].clone())
        .ok_or("points parameter must be list of float atoms (of the form `(a . b)`)")?;
    let sample_count = try_get_int(&params[1])
        .ok_or("sample-count parameter must be an int")?;

    if !are_points_sorted(&points) {
        return Err("Each point must have an increasing x value".to_owned());
    } else if points.len() < 2 {
        return Err("points parameter must be a list of at least 2 values".to_owned());
    }
    // TODO: Use a cubic bezier instead
    let keys = points.iter().copied().map(|(x, y)| {
        Key::new(x, y, Interpolation::Cosine)
    }).collect::<Vec<_>>();
    let spline = Spline::from_vec(keys);

    let min_x = points[0].0;
    let max_x = points.last().unwrap().0;

    let mut data = vec![];
    for index in 0..sample_count as usize {
        let t = min_x + ((index as f64) / (sample_count as f64)) * (max_x - min_x);
        let x = spline.sample(t).unwrap();
        data.push(x);
    }
    Ok(Rc::new(Value::WaveData(data)))
}

fn wd_low_pass(ctx: &RunContext, params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    use biquad::*;
    let sr = ctx.scope.borrow().lookup("wd-sample-rate");
    // # of samples per second
    let sample_rate = try_get_int(&sr)
        .ok_or("wd-sample-rate must be globally set as an int")?;
    let sample_freq = 5.0 / sample_rate as f64;
    let cutoff_freq = try_get_float(&params[0])
        .ok_or("cutoff-freq parameter must be a float")?;
    let wavedata = try_get_wavedata(&params[1])
        .ok_or("wavedata parameter must be a wavedata object")?;

    let coeffs = Coefficients::<f64>::from_params(
        Type::LowPass,
        sample_freq.hz(),
        cutoff_freq.hz(),
        Q_BUTTERWORTH_F64
    ).unwrap();
    let mut bq = DirectForm2Transposed::<f64>::new(coeffs);
    let data = wavedata.iter().map(|&v| bq.run(v)).collect::<Vec<_>>();
    Ok(Rc::new(Value::WaveData(data)))
}

fn gaussian(x: f64, variance: f64) -> f64 {
    let normalizing_constant = 1.0 / (variance * (2.0 * PI).sqrt());
    let exp_param = -(x.powi(2)) / (2.0 * variance.powi(2));
    normalizing_constant * exp_param.exp()
}

fn wd_gaussian_blur(params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let variance = try_get_float(&params[0])
        .ok_or("variance parameter must be a float")?;
    let wavedata = try_get_wavedata(&params[1])
        .ok_or("wavedata parameter must be a wavedata object")?;
    if variance <= 0.0 {
        return Err("Variance must be > 0.0".to_owned());
    }
    // Compute gaussian mask
    let epsilon = 1e-9f64; // Don't bother considering below this value
    let mut mask = vec![];
    for i in 0..wavedata.len() {
        let g = gaussian(i as f64, variance);
        if g < epsilon {
            break;
        }
        mask.push(g);
    }

    let mut data = Vec::with_capacity(wavedata.len());
    for i in 0..wavedata.len() {
        let mut acc = 0.0;
        for j in -(mask.len() as i64 - 1)..(mask.len() as i64) {
            let idx = i as i64 + j;
            if idx >= 0 && idx < wavedata.len() as i64 {
                let m = mask[j.abs() as usize];
                acc += m * wavedata[idx as usize];
            }
        }
        data.push(acc);
    }
    Ok(Rc::new(Value::WaveData(data)))
}

fn wd_save(ctx: &RunContext, params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let sr = ctx.scope.borrow().lookup("wd-sample-rate");
    let sample_rate = try_get_int(&sr)
        .ok_or("wd-sample-rate must be globally set as an int")?;
    let wavedata = try_get_wavedata(&params[0])
        .ok_or("wavedata parameter must be a wavedata object")?;
    let file_path = try_get_string(&params[1])
        .ok_or("file-path parameter must be a String")?;

    if save_wave(sample_rate as u32, &wavedata, &file_path).is_ok() {
        Ok(Value::nil())
    } else {
        Err(format!("Failed saving wave file to {file_path}"))
    }
}

fn wd_play(ctx: &RunContext, params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let sr = ctx.scope.borrow().lookup("wd-sample-rate");
    let sample_rate = try_get_int(&sr)
        .ok_or("wd-sample-rate must be globally set as an int")?;
    let wavedata = try_get_wavedata(&params[0])
        .ok_or("wavedata parameter must be a wavedata object")?;

    if play_wave(sample_rate as u32, &wavedata).is_ok() {
        Ok(Value::nil())
    } else {
        Err("Failed playing wavedata".to_owned())
    }
}

fn wd_multiply(params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let lhs = try_get_wavedata(&params[0])
        .ok_or("lhs parameter must be a wavedata object")?;
    let rhs = try_get_wavedata(&params[1])
        .ok_or("rhs parameter must be a wavedata object")?;
    if lhs.len() != rhs.len() {
        return Err("wd-multiply must be called on two equally sized wavedata objects".to_owned());
    }
    let mut data = lhs;
    for i in 0..data.len() {
        data[i] *= rhs[i];
    }
    Ok(Rc::new(Value::WaveData(data)))
}

fn wd_superimpose(params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let lhs = try_get_wavedata(&params[0])
        .ok_or("lhs parameter must be a wavedata object")?;
    let rhs = try_get_wavedata(&params[1])
        .ok_or("rhs parameter must be a wavedata object")?;
    let (mut data, to_add) = if lhs.len() > rhs.len() {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    for i in 0..to_add.len() {
        data[i] += to_add[i];
    }
    Ok(Rc::new(Value::WaveData(data)))
}

fn wd_superimpose_insert(params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let from_data = try_get_wavedata(&params[0])
        .ok_or("`from` parameter must be a wavedata object")?;
    let index = try_get_int(&params[1])
        .ok_or("`index` parameter must be an int")?;
    let mut data = try_get_wavedata(&params[2])
        .ok_or("`to` parameter must be a wavedata object")?;

    let padding_amount = (index - data.len() as i64).max(0) as usize;
    data.resize(data.len() + padding_amount, 0.0);
    for (i, n) in from_data.iter().enumerate() {
        let offset = i + index as usize;
        if offset >= data.len() {
            data.push(*n);
        } else {
            data[offset] += n;
        }
    }
    Ok(Rc::new(Value::WaveData(data)))
}

fn wd_len(params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let param = try_get_wavedata(&params[0])
        .ok_or("wd-len must be called on a single wavedata object")?;
    Ok(Rc::new(Value::Int(param.len() as i64)))
}

fn wd_concat(params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let mut lhs = try_get_wavedata(&params[0])
        .ok_or("lhs parameter must be a wavedata object")?;
    let rhs = try_get_wavedata(&params[1])
        .ok_or("rhs parameter must be a wavedata object")?;
    lhs.extend_from_slice(&rhs);
    Ok(Rc::new(Value::WaveData(lhs)))
}

fn wd_noise(params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let mut rng = thread_rng();
    let sample_count = try_get_int(&params[0])
        .ok_or("sample-count parameter must be an int")?;
    let mut data = vec![];
    for _ in 0..sample_count {
        data.push(rng.gen_range(-1.0..1.0));
    }
    Ok(Rc::new(Value::WaveData(data)))
}

fn wd_subsample(params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let start = try_get_int(&params[0])
        .ok_or("start parameter must be an int")?;
    let end = try_get_int(&params[1])
        .ok_or("end parameter must be an int")?;
    let wavedata = try_get_wavedata(&params[2])
        .ok_or("data parameter must be a wavedata object")?;

    if start < 0 || start as usize > wavedata.len() {
        return Err("start parameter is out of bounds".to_owned());
    }
    if end < 0 || end as usize > wavedata.len() {
        return Err("end parameter is out of bounds".to_owned());
    }
    if start > end {
        return Err("start must be greater than end".to_owned());
    }

    let data = wavedata[start as usize..end as usize].to_vec();
    Ok(Rc::new(Value::WaveData(data)))
}

fn wd_reverse(params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    let mut data = try_get_wavedata(&params[0])
        .ok_or("data parameter must be a wavedata object")?;
    data.reverse();
    Ok(Rc::new(Value::WaveData(data)))
}

fn char_list_as_string(value: Rc<Value>) -> RunResult<Rc<Value>> {
    let mut s = String::new();
    let mut head = value;
    loop {
        match &*head {
            Value::Cons(car, cdr) => {
                if let Some(c) = try_get_char(car) {
                    s.push(c);
                } else {
                    return Err("Invalid character list".to_owned());
                }
                head = cdr.clone();
            },
            Value::Builtin(Builtin::Nil) => break,
            _ => return Err("Invalid character list".to_owned()),
        }
    }
    Ok(Rc::new(Value::String(s)))
}

fn string_as_char_list(value: &Value) -> RunResult<Rc<Value>> {
    match value {
        Value::String(s) => {
            Ok(s.chars().rev().fold(Value::nil(), |acc, it| {
                Rc::new(Value::Cons(Rc::new(Value::Int(it as i64)), acc))
            }))
        },
        _ => Err(format!("Value is not a string: `{value:?}`"))
    }
}

fn sexpr_as_string(value: &Value) -> Rc<Value> {
    Rc::new(Value::String(format!("{value:?}")))
}

fn is_truthy(value: Rc<Value>) -> bool {
    match &*value {
        Value::Int(value)   => *value != 0,
        Value::Float(value) => *value != 0.0,
        Value::Builtin(Builtin::Nil) => false,
        _                   => true,
    }
}

fn get_car(value: &Value) -> Option<Rc<Value>> {
    match value {
        Value::Cons(car, _) => Some(car.clone()),
        Value::Builtin(Builtin::Nil) => Some(Value::nil()),
        _ => None,
    }
}

fn get_cdr(value: &Value) -> Option<Rc<Value>> {
    match value {
        Value::Cons(_, cdr) => Some(cdr.clone()),
        Value::Builtin(Builtin::Nil) => Some(Value::nil()),
        _ => None,
    }
}

fn try_get_char(expr: &Value) -> Option<char> {
    try_get_int(expr).and_then(|value| char::from_u32(value as u32))
}

fn try_get_int(value: &Value) -> Option<i64> {
    match value {
        Value::Int(value) => Some(*value),
        _ => None,
    }
}

fn try_get_float(value: &Value) -> Option<f64> {
    match value {
        Value::Float(value) => Some(*value),
        _ => None,
    }
}

fn try_get_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        _ => None,
    }
}

fn try_get_wavedata(value: &Value) -> Option<Vec<f64>> {
    match value {
        Value::WaveData(value) => Some(value.clone()),
        _ => None,
    }
}

fn try_get_list(root: Rc<Value>) -> Option<Vec<Rc<Value>>> {
    let mut elems = vec![];
    let mut value = root;
    while let Value::Cons(car, cdr) = &*value {
        elems.push(car.clone());
        value = cdr.clone();
    }
    Some(elems)
}

fn try_get_point_list(root: Rc<Value>) -> Option<Vec<(f64, f64)>> {
    let values = try_get_list(root)?;
    let mut points = vec![];
    for value in values.iter() {
        let (x_expr, y_expr) = try_get_cons_pair(value)?;
        let x = try_get_float(&x_expr)?;
        let y = try_get_float(&y_expr)?;
        points.push((x, y));
    }
    Some(points)
}

fn try_get_cons_pair(pair: &Value) -> Option<(Rc<Value>, Rc<Value>)> {
    match pair {
        Value::Cons(car, cdr) => Some((car.clone(), cdr.clone())),
        _ => None,
    }
}

fn eval_lambda(ctx: &RunContext, params: &[Rc<SExpr>]) -> RunResult<Rc<Value>> {
    param_count_ge(Builtin::Lambda, params, 2)?;
    let arg_exprs = params[0].clone();

    let mut arg_names = vec![];
    for arg in unfold(arg_exprs).iter() {
        if let Some(name) = get_expr_symbol_name(arg) {
            arg_names.push(name.clone());
        } else {
            return Err("The first param of a function definition must by a symbol list".to_owned());
        }
    }
    // Inefficient but simple. It would be nice to deal with this at parse time
    let referenced_vars = params[1..].iter()
        .flat_map(|expr| find_symbols(&expr))
        .filter(|symbol| {
            !arg_names.contains(symbol)
        })
        .collect::<Vec<_>>();
    let mut captured_scope = HashMap::new();
    for var in referenced_vars.iter() {
        captured_scope.insert(var.clone(), ctx.scope.borrow().lookup(&var));
    }
    let f = Value::Function(captured_scope, arg_names, params[1..].to_vec());
    Ok(Rc::new(f))
}

fn find_symbols(expr: &SExpr) -> Vec<String> {
    match expr {
        SExpr::Atom(value) => match &**value {
            Value::Symbol(name) => vec![name.clone()],
            _ => vec![],
        },
        SExpr::S(car, cdr) => {
            let mut symbols = find_symbols(car);
            symbols.extend_from_slice(&find_symbols(cdr));
            symbols
        },
    }
}

fn eval_set(scope: Rc<RefCell<Scope>>, params: &[Rc<Value>]) -> RunResult<Rc<Value>> {
    param_count_eq(Builtin::Set, params, 2)?;
    if let Some(name) = get_symbol_name(&params[0]) {
        let set_value = params[1].clone();
        scope.borrow_mut().insert(&name, set_value.clone());
        Ok(set_value)
    } else {
        Err("The first param to `set` must be a symbol".to_owned())
    }
}

fn get_expr_symbol_name(sexpr: &SExpr) -> Option<String> {
    match sexpr.atom_value() {
        Some(atom) => {
            match &*atom {
                Value::Symbol(name) => Some(name.clone()),
                _ => None,
            }
        },
        None => None,
    }
}

fn get_symbol_name(value: &Value) -> Option<String> {
    match value {
        Value::Symbol(name) => Some(name.clone()),
        _ => None,
    }
}

// Assumes the given builtin is a valid arithmetic op
fn eval_arithmetic(builtin: Builtin, lhs: &Value, rhs: &Value) -> RunResult<Rc<Value>> {
    match (lhs, rhs) {
        (Value::Int(lhs), Value::Int(rhs))     => Ok(eval_arithmetic_int(builtin, *lhs, *rhs)),
        (Value::Int(lhs), Value::Float(rhs))   => Ok(eval_arithmetic_float(builtin, *lhs as f64, *rhs)),
        (Value::Float(lhs), Value::Int(rhs))   => Ok(eval_arithmetic_float(builtin, *lhs, *rhs as f64)),
        (Value::Float(lhs), Value::Float(rhs)) => Ok(eval_arithmetic_float(builtin, *lhs, *rhs)),
        (lhs, rhs) if lhs.is_nil() || rhs.is_nil() => {
            Err("Cannot perform arithmetic on nil".to_owned())
        },
        _ => unreachable!()
    }
}

fn eval_arithmetic_int(builtin: Builtin, lhs: i64, rhs: i64) -> Rc<Value> {
    Rc::new(Value::Int(match builtin {
        Builtin::Add => lhs + rhs,
        Builtin::Sub => lhs - rhs,
        Builtin::Mul => lhs * rhs,
        Builtin::Div => lhs / rhs,
        Builtin::Mod => lhs % rhs,
        Builtin::Pow => lhs.pow(rhs as u32),
        _ => unreachable!()
    }))
}

fn eval_arithmetic_float(builtin: Builtin, lhs: f64, rhs: f64) -> Rc<Value> {
    Rc::new(Value::Float(match builtin {
        Builtin::Add => lhs + rhs,
        Builtin::Sub => lhs - rhs,
        Builtin::Mul => lhs * rhs,
        Builtin::Div => lhs / rhs,
        Builtin::Mod => lhs % rhs,
        Builtin::Pow => lhs.powf(rhs),
        _ => unreachable!()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Returns the last evaluated expr
    fn eval_str(s: &str) -> Rc<Value> {
        let exprs = parse_str(s).unwrap()
            .into_iter()
            .collect::<Vec<_>>();
        let mut ctx = RunContext::new();
        eval_progn(&mut ctx, &exprs).unwrap()
    }

    #[test]
    fn test_eval_arithmetic() {
        assert_eq!("4", format!("{:?}", eval_str("(+ 1 3)")));
        assert_eq!("-2", format!("{:?}", eval_str("(- 1 3)")));
        assert_eq!("17", format!("{:?}", eval_str("(+ 1 (+ 7 9))")));
        assert_eq!("2", format!("{:?}", eval_str("(/ 5 2)")));
        assert_eq!("8", format!("{:?}", eval_str("(* 4 2)")));
        assert_eq!("0", format!("{:?}", eval_str("(/ 1 (* 7 9))")));
        assert_eq!("0.0", format!("{:?}", eval_str("(* 0.0 7)")));
        assert_eq!("0.125", format!("{:?}", eval_str("(/ 1 (* 4.0 2))")));
        assert_eq!("2", format!("{:?}", eval_str("(% 42 5)")));
    }

    #[test]
    fn test_eval_progn() {
        assert_eq!("2", format!("{:?}", eval_str("(progn (% 42 5))")));
        assert_eq!("2", format!("{:?}", eval_str("(progn 6 nil (+ 4 2) (% 42 5))")));
        assert_eq!("nil", format!("{:?}", eval_str("(progn nil)")));
    }

    #[test]
    fn test_eval_set() {
        assert_eq!("8", format!("{:?}", eval_str("(set 'x 4) (+ x x)")));
    }

    #[test]
    fn test_eval_fun() {
        assert_eq!("3", format!("{:?}", eval_str("(defun add (a b) (+ a b)) (add 1 2)")));
        assert_eq!("3", format!("{:?}", eval_str("((lambda (a b) (+ a b)) 1 2)")));
    }

    #[test]
    fn test_eval_if() {
        assert_eq!("2", format!("{:?}", eval_str("(if nil 1 2)")));
    }

    #[test]
    fn test_eval_false() {
        assert_eq!("1", format!("{:?}", eval_str("(false? nil)")));
        assert_eq!("1", format!("{:?}", eval_str("(false? 'nil)")));
        assert_eq!("1", format!("{:?}", eval_str("(false? '())")));
        assert_eq!("nil", format!("{:?}", eval_str("(false? 42)")));
        assert_eq!("nil", format!("{:?}", eval_str("(false? '(4))")));
    }

    #[test]
    fn test_eval_eq() {
        assert_eq!("1", format!("{:?}", eval_str("(eq? '(2 3 4) '(2 3 4))")));
        assert_eq!("nil", format!("{:?}", eval_str("(eq? '(2 3 4) '(2 3 5))")));
        assert_eq!("nil", format!("{:?}", eval_str("(eq? '(2 3 4) 4)")));
    }

    #[test]
    fn test_eval_quote() {
        assert_eq!("2", format!("{:?}", eval_str("'2")));
        assert_eq!("a", format!("{:?}", eval_str("'a")));
        assert_eq!("(2 . nil)", format!("{:?}", eval_str("'(2)")));
    }

    #[test]
    fn test_eval_cons() {
        assert_eq!("(1 . (2 . nil))", format!("{:?}", eval_str("(cons 1 '(2))")));
        assert_eq!("(1 . nil)", format!("{:?}", eval_str("(cons 1 (list))")));
        assert_eq!("(1 . nil)", format!("{:?}", eval_str("(cons 1 nil)")));
    }
}
